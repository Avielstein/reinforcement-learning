import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .networks import ActorNetwork, CriticNetwork
from .curiosity_module import IntrinsicCuriosityModule
from .memory import PPOMemory

class PPOCuriousAgent:
    """
    PPO Agent with Intrinsic Curiosity Module.
    
    Combines PPO algorithm with curiosity-driven exploration for fish swimming.
    """
    
    def __init__(self, state_dim=152, action_dim=4, hidden_dim=256,
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 entropy_coef=0.01, value_coef=0.5, curiosity_weight=0.1,
                 buffer_size=2048, batch_size=64, n_epochs=10,
                 curiosity_lr=1e-3, feature_dim=64, device=None):
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.curiosity_weight = curiosity_weight
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Curiosity module
        self.curiosity_module = IntrinsicCuriosityModule(
            state_dim, action_dim, feature_dim, hidden_dim, 
            curiosity_lr, device=self.device
        )
        
        # Memory buffer
        self.memory = PPOMemory(buffer_size, state_dim, action_dim, self.device)
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.total_reward_history = []
        self.intrinsic_reward_history = []
        self.loss_history = []
        
        # Current state for continuous interaction
        self.current_state = None
        
    def get_action(self, state, training=True):
        """
        Get action from the current policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: State value estimate
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get action from actor
            if training:
                action, log_prob = self.actor.get_action_and_log_prob(state)
            else:
                # For evaluation, use mean action (no sampling)
                mean, _ = self.actor(state)
                action = mean
                log_prob = torch.zeros(1).to(self.device)
            
            # Get value from critic
            value = self.critic(state)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in memory with curiosity reward."""
        # Compute intrinsic reward
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        intrinsic_reward = self.curiosity_module.compute_intrinsic_reward(
            state_tensor, action_tensor, next_state_tensor
        ).cpu().numpy()[0]
        
        # Get value and log prob for the stored state
        with torch.no_grad():
            value = self.critic(state_tensor).cpu().numpy()[0][0]
            log_prob = self.actor.get_log_prob(state_tensor, action_tensor).cpu().numpy()[0]
        
        # Store in memory
        self.memory.store(state, action, reward, intrinsic_reward, value, log_prob, done, next_state)
    
    def update(self):
        """Update the agent using collected experience."""
        if not self.memory.is_full():
            return {}
        
        # Get the last state value for GAE computation
        last_state = torch.FloatTensor(self.memory.next_states[self.memory.size-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.critic(last_state).cpu().numpy()[0][0]
        
        # Compute advantages and returns
        advantages, returns = self.memory.compute_gae(
            next_value, self.gamma, self.gae_lambda, self.curiosity_weight
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get all data from memory
        data = self.memory.get_all_data()
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        next_states = torch.FloatTensor(data['next_states']).to(self.device)
        
        # Training metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        
        # PPO update epochs
        for epoch in range(self.n_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Actor update
                new_log_probs = self.actor.get_log_prob(batch_states, batch_actions)
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                mean, log_std = self.actor(batch_states)
                entropy = (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1).mean()
                entropy_loss = -self.entropy_coef * entropy
                
                # Total actor loss
                total_actor_loss_batch = actor_loss + entropy_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Critic update
                values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                # Accumulate losses
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Update curiosity module
        curiosity_metrics = self.curiosity_module.update(states, actions, next_states)
        
        # Clear memory
        self.memory.clear()
        
        # Update training statistics
        self.training_step += 1
        
        # Compile metrics
        metrics = {
            'actor_loss': total_actor_loss / (self.n_epochs * (len(states) // self.batch_size + 1)),
            'critic_loss': total_critic_loss / (self.n_epochs * (len(states) // self.batch_size + 1)),
            'entropy_loss': total_entropy_loss / (self.n_epochs * (len(states) // self.batch_size + 1)),
            'avg_advantage': advantages.mean(),
            'avg_return': returns.mean(),
            'training_step': self.training_step,
            **curiosity_metrics
        }
        
        self.loss_history.append(metrics)
        
        return metrics
    
    def update_config(self, config):
        """Update agent configuration (for live parameter tuning)."""
        if 'learning_rate' in config:
            self.learning_rate = config['learning_rate']
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        if 'gamma' in config:
            self.gamma = config['gamma']
        if 'gae_lambda' in config:
            self.gae_lambda = config['gae_lambda']
        if 'clip_range' in config:
            self.clip_range = config['clip_range']
        if 'entropy_coef' in config:
            self.entropy_coef = config['entropy_coef']
        if 'value_coef' in config:
            self.value_coef = config['value_coef']
        if 'curiosity_weight' in config:
            self.curiosity_weight = config['curiosity_weight']
    
    def get_statistics(self):
        """Get comprehensive agent statistics."""
        memory_stats = self.memory.get_statistics()
        curiosity_stats = self.curiosity_module.get_statistics()
        
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'avg_total_reward': np.mean(self.total_reward_history[-100:]) if self.total_reward_history else 0.0,
            'avg_intrinsic_reward': np.mean(self.intrinsic_reward_history[-100:]) if self.intrinsic_reward_history else 0.0,
            'recent_actor_loss': self.loss_history[-1]['actor_loss'] if self.loss_history else 0.0,
            'recent_critic_loss': self.loss_history[-1]['critic_loss'] if self.loss_history else 0.0,
            **memory_stats,
            **curiosity_stats
        }
    
    def save(self, filepath):
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'total_reward_history': self.total_reward_history,
            'intrinsic_reward_history': self.intrinsic_reward_history,
            'loss_history': self.loss_history,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'curiosity_weight': self.curiosity_weight
            }
        }, filepath)
        
        # Save curiosity module separately
        curiosity_filepath = filepath.replace('.pt', '_curiosity.pt')
        self.curiosity_module.save(curiosity_filepath)
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.total_reward_history = checkpoint['total_reward_history']
        self.intrinsic_reward_history = checkpoint['intrinsic_reward_history']
        self.loss_history = checkpoint['loss_history']
        
        # Load curiosity module
        curiosity_filepath = filepath.replace('.pt', '_curiosity.pt')
        try:
            self.curiosity_module.load(curiosity_filepath)
        except FileNotFoundError:
            print(f"Warning: Curiosity module file {curiosity_filepath} not found")
    
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_count += 1
        self.current_state = None

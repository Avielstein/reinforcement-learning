"""
TD Fish Agent - Main agent combining policy and TD learning.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque

from ..config.environment import EnvironmentConfig
from ..config.td_config import TDConfig
from ..models.policy_net import ContinuousPolicyNetwork
from .td_learner import TDLearner, MultiStepTDLearner, EnsembleTDLearner


class TDFishAgent:
    """Main TD learning agent for fish following task."""
    
    def __init__(self, env_config: EnvironmentConfig, td_config: TDConfig, device: str = 'cpu'):
        self.env_config = env_config
        self.td_config = td_config
        self.device = device
        
        # Create policy network
        self.policy = ContinuousPolicyNetwork(
            obs_dim=env_config.observation_dim,
            action_dim=env_config.action_dim,
            hidden_dims=td_config.policy_network_hidden,
            activation='relu'
        ).to(device)
        
        # Create TD learner
        if td_config.method == 'n_step_td':
            self.td_learner = MultiStepTDLearner(
                env_config.observation_dim, td_config, device
            )
        else:
            self.td_learner = TDLearner(
                env_config.observation_dim, td_config, device
            )
        
        # Policy optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=td_config.policy_lr
        )
        
        # Exploration parameters
        self.exploration_params = td_config.get_exploration_params()
        self.current_epsilon = self.exploration_params.get('epsilon_start', 1.0)
        self.current_noise_std = self.exploration_params.get('noise_std', 0.1)
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_distances = deque(maxlen=100)
        
        # Performance tracking
        self.best_avg_reward = float('-inf')
        self.best_avg_distance = float('inf')
        
        # Action smoothing
        self.previous_action = None
        
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using policy with exploration."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.td_config.exploration_method == 'epsilon_greedy':
                if not deterministic and np.random.random() < self.current_epsilon:
                    # Random action
                    action = np.random.uniform(-1, 1, self.env_config.action_dim)
                else:
                    # Policy action
                    action, _ = self.policy.sample(obs_tensor, deterministic=True)
                    action = action.cpu().numpy().flatten()
            
            elif self.td_config.exploration_method == 'gaussian_noise':
                # Policy action with Gaussian noise
                action, _ = self.policy.sample(obs_tensor, deterministic=deterministic)
                action = action.cpu().numpy().flatten()
                
                if not deterministic:
                    noise = np.random.normal(0, self.current_noise_std, action.shape)
                    action = action + noise
            
            else:
                # Default: sample from policy
                action, _ = self.policy.sample(obs_tensor, deterministic=deterministic)
                action = action.cpu().numpy().flatten()
        
        # Apply action smoothing
        if self.previous_action is not None:
            action = (self.env_config.action_smoothing * self.previous_action + 
                     (1 - self.env_config.action_smoothing) * action)
        
        # Clamp to valid range
        action = np.clip(action, -1, 1)
        self.previous_action = action.copy()
        
        return action
    
    def update(self, obs: np.ndarray, action: np.ndarray, reward: float,
               next_obs: np.ndarray, done: bool) -> Dict[str, float]:
        """Update agent with experience."""
        # Update TD learner
        td_stats = self.td_learner.update(obs, action, reward, next_obs, done)
        
        # Update policy (Actor-Critic style)
        policy_stats = self._update_policy(obs, action, reward, next_obs, done)
        
        # Combine statistics
        stats = {**td_stats, **policy_stats}
        
        # Update exploration parameters
        self._update_exploration()
        
        # Update step count
        self.total_steps += 1
        
        return stats
    
    def _update_policy(self, obs: np.ndarray, action: np.ndarray, reward: float,
                      next_obs: np.ndarray, done: bool) -> Dict[str, float]:
        """Update policy using TD value estimates."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)
        
        # Get current and next values from TD learner
        current_value = self.td_learner.get_value(obs)
        next_value = self.td_learner.get_value(next_obs) if not done else 0.0
        
        # Calculate advantage
        td_target = reward + self.td_config.gamma * next_value * (1 - done)
        advantage = td_target - current_value
        
        # Policy gradient update
        log_prob = self.policy.log_prob(obs_tensor, action_tensor)
        entropy = self.policy.entropy(obs_tensor)
        
        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_prob * advantage).mean()
        
        # Add entropy regularization
        entropy_loss = -self.td_config.entropy_coef * entropy.mean()
        
        total_policy_loss = policy_loss + entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        
        # Gradient clipping
        if self.td_config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                         self.td_config.gradient_clip_norm)
        
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.mean().item(),
            'advantage': advantage,
            'log_prob': log_prob.mean().item()
        }
    
    def _update_exploration(self):
        """Update exploration parameters."""
        if self.td_config.exploration_method == 'epsilon_greedy':
            self.current_epsilon = max(
                self.exploration_params['epsilon_end'],
                self.current_epsilon * self.exploration_params['epsilon_decay']
            )
        elif self.td_config.exploration_method == 'gaussian_noise':
            self.current_noise_std = max(
                0.01,  # Minimum noise
                self.current_noise_std * self.exploration_params['noise_decay']
            )
    
    def end_episode(self, episode_reward: float, avg_distance: float):
        """Called at the end of each episode."""
        # Reset for new episode
        self.td_learner.reset_episode()
        self.policy.reset_action_smoothing()
        self.previous_action = None
        
        # Update episode statistics
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.episode_distances.append(avg_distance)
        
        # Update best performance
        if len(self.episode_rewards) >= 10:
            avg_reward = np.mean(list(self.episode_rewards)[-10:])
            avg_dist = np.mean(list(self.episode_distances)[-10:])
            
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
            
            if avg_dist < self.best_avg_distance:
                self.best_avg_distance = avg_dist
        
        # Decay learning rates
        if self.episode_count % 100 == 0:
            self.td_learner.decay_learning_rate()
            
            # Decay policy learning rate
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] *= self.td_config.lr_decay
                param_group['lr'] = max(param_group['lr'], self.td_config.min_lr)
    
    def get_value(self, observation: np.ndarray) -> float:
        """Get value estimate for observation."""
        return self.td_learner.get_value(observation)
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance."""
        episode_rewards = []
        episode_distances = []
        success_count = 0
        
        for _ in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            distances = []
            done = False
            
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                distances.append(info['distance_to_target'])
            
            episode_rewards.append(episode_reward)
            avg_distance = np.mean(distances)
            episode_distances.append(avg_distance)
            
            if avg_distance < env.config.success_distance:
                success_count += 1
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_distance': np.mean(episode_distances),
            'std_distance': np.std(episode_distances),
            'success_rate': success_count / num_episodes,
            'min_distance': np.min(episode_distances),
            'max_reward': np.max(episode_rewards)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        stats = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'current_epsilon': self.current_epsilon,
            'current_noise_std': self.current_noise_std,
            'best_avg_reward': self.best_avg_reward,
            'best_avg_distance': self.best_avg_distance,
            'policy_lr': self.policy_optimizer.param_groups[0]['lr']
        }
        
        # Add recent performance
        if self.episode_rewards:
            stats.update({
                'recent_avg_reward': np.mean(list(self.episode_rewards)[-10:]),
                'recent_avg_distance': np.mean(list(self.episode_distances)[-10:]),
                'reward_trend': np.mean(list(self.episode_rewards)[-5:]) - np.mean(list(self.episode_rewards)[-10:-5]) if len(self.episode_rewards) >= 10 else 0
            })
        
        # Add TD learner stats
        stats.update(self.td_learner.get_stats())
        
        return stats
    
    def save(self, filepath: str):
        """Save agent state."""
        state = {
            'policy_state_dict': self.policy.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'current_epsilon': self.current_epsilon,
            'current_noise_std': self.current_noise_std,
            'best_avg_reward': self.best_avg_reward,
            'best_avg_distance': self.best_avg_distance,
            'episode_rewards': list(self.episode_rewards),
            'episode_distances': list(self.episode_distances),
            'env_config': self.env_config,
            'td_config': self.td_config
        }
        
        # Save TD learner separately
        td_filepath = filepath.replace('.pt', '_td_learner.pt')
        self.td_learner.save(td_filepath)
        
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        state = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(state['policy_state_dict'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer_state_dict'])
        self.episode_count = state['episode_count']
        self.total_steps = state['total_steps']
        self.current_epsilon = state['current_epsilon']
        self.current_noise_std = state['current_noise_std']
        self.best_avg_reward = state['best_avg_reward']
        self.best_avg_distance = state['best_avg_distance']
        self.episode_rewards = deque(state['episode_rewards'], maxlen=100)
        self.episode_distances = deque(state['episode_distances'], maxlen=100)
        
        # Load TD learner
        td_filepath = filepath.replace('.pt', '_td_learner.pt')
        self.td_learner.load(td_filepath)
    
    def set_training_mode(self, training: bool = True):
        """Set training mode for networks."""
        self.policy.train(training)
        self.td_learner.critic.train(training)
    
    def get_action_distribution(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get action distribution parameters (mean, std)."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, std = self.policy(obs_tensor)
            return mean.cpu().numpy().flatten(), std.cpu().numpy().flatten()


class EnsembleTDFishAgent(TDFishAgent):
    """TD Fish Agent with ensemble of TD learners for uncertainty estimation."""
    
    def __init__(self, env_config: EnvironmentConfig, td_config: TDConfig, 
                 num_learners: int = 3, device: str = 'cpu'):
        # Initialize base agent but replace TD learner with ensemble
        super().__init__(env_config, td_config, device)
        
        # Replace single TD learner with ensemble
        self.td_learner = EnsembleTDLearner(
            env_config.observation_dim, td_config, num_learners, device
        )
        
        self.num_learners = num_learners
    
    def get_value_with_uncertainty(self, observation: np.ndarray) -> Tuple[float, float]:
        """Get value estimate with uncertainty."""
        return self.td_learner.get_value(observation)
    
    def _update_policy(self, obs: np.ndarray, action: np.ndarray, reward: float,
                      next_obs: np.ndarray, done: bool) -> Dict[str, float]:
        """Update policy using ensemble value estimates."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        # Get value estimates with uncertainty
        current_value, current_uncertainty = self.td_learner.get_value(obs)
        next_value, next_uncertainty = self.td_learner.get_value(next_obs) if not done else (0.0, 0.0)
        
        # Calculate advantage
        td_target = reward + self.td_config.gamma * next_value * (1 - done)
        advantage = td_target - current_value
        
        # Uncertainty-weighted policy update
        uncertainty_weight = 1.0 / (1.0 + current_uncertainty)  # Lower weight for uncertain states
        
        # Policy gradient update
        log_prob = self.policy.log_prob(obs_tensor, action_tensor)
        entropy = self.policy.entropy(obs_tensor)
        
        # Weighted policy loss
        policy_loss = -(log_prob * advantage * uncertainty_weight).mean()
        entropy_loss = -self.td_config.entropy_coef * entropy.mean()
        
        total_policy_loss = policy_loss + entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        
        if self.td_config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                         self.td_config.gradient_clip_norm)
        
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.mean().item(),
            'advantage': advantage,
            'log_prob': log_prob.mean().item(),
            'value_uncertainty': current_uncertainty,
            'uncertainty_weight': uncertainty_weight
        }

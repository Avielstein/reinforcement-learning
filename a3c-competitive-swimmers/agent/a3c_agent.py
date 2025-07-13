import torch
import torch.nn.functional as F
import numpy as np
import threading
import time
from typing import List, Tuple, Dict, Optional
from collections import deque

from .networks import TrustRegionActorCritic, SharedGlobalNetwork


class A3CWorker:
    """
    Individual A3C worker that runs in its own thread.
    
    Each worker:
    1. Interacts with its own environment copy
    2. Collects experiences using local policy
    3. Computes gradients and updates global network
    4. Syncs with global network periodically
    """
    
    def __init__(self, 
                 worker_id: int,
                 global_network: SharedGlobalNetwork,
                 env_factory,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 max_steps: int = 5,
                 trust_region_coef: float = 0.01,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        
        self.worker_id = worker_id
        self.global_network = global_network
        self.env = env_factory()
        
        # Create local network
        self.local_network = TrustRegionActorCritic(
            state_dim, action_dim, max_kl=trust_region_coef
        )
        
        # Optimizer for local network
        self.optimizer = torch.optim.Adam(
            self.local_network.parameters(), lr=learning_rate
        )
        
        # Hyperparameters
        self.gamma = gamma
        self.max_steps = max_steps
        self.trust_region_coef = trust_region_coef
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_steps = 0
        self.episodes_completed = 0
        
        # Trust region statistics
        self.kl_divergences = deque(maxlen=100)
        self.successful_updates = 0
        self.total_updates = 0
        
        # Threading
        self.running = False
        self.thread = None
    
    def sync_with_global(self):
        """Sync local network parameters with global network."""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def compute_returns_and_advantages(self, next_value: float) -> Tuple[List[float], List[float]]:
        """
        Compute discounted returns and advantage estimates.
        
        Args:
            next_value: Value estimate for the next state
            
        Returns:
            returns: Discounted returns
            advantages: Advantage estimates (returns - values)
        """
        returns = []
        advantages = []
        
        # Compute returns using bootstrapped value
        R = next_value
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.gamma * R
            returns.insert(0, R)
        
        # Compute advantages
        for i in range(len(returns)):
            advantage = returns[i] - self.values[i]
            advantages.append(advantage)
        
        return returns, advantages
    
    def update_global_network(self, next_value: float) -> Dict[str, float]:
        """
        Update global network using collected experiences.
        
        Args:
            next_value: Value estimate for next state
            
        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.states) == 0:
            return {}
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.stack(self.values)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old policy logits for trust region calculation
        with torch.no_grad():
            old_logits, _ = self.local_network.forward(states)
        
        # Compute losses
        new_logits, new_values = self.local_network.forward(states)
        
        # Policy loss
        new_dist = torch.distributions.Categorical(logits=new_logits)
        new_log_probs = new_dist.log_prob(actions)
        entropy = new_dist.entropy().mean()
        
        policy_loss = -(new_log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values.squeeze(), returns)
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coef * value_loss - 
                     self.entropy_coef * entropy)
        
        # Check trust region constraint
        kl_div = self.local_network.compute_kl_divergence(old_logits, new_logits)
        
        update_applied = True
        if kl_div.item() > self.trust_region_coef:
            # Skip update if KL divergence is too high
            update_applied = False
        else:
            # Apply gradients to global network
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), max_norm=0.5)
            
            # Copy gradients to global network
            for local_param, global_param in zip(
                self.local_network.parameters(), 
                self.global_network.parameters()
            ):
                if local_param.grad is not None:
                    global_param.grad = local_param.grad.clone()
            
            # Update global network
            global_optimizer = torch.optim.Adam(self.global_network.parameters())
            global_optimizer.step()
            global_optimizer.zero_grad()
            
            # Sync local network with updated global network
            self.sync_with_global()
        
        # Track statistics
        self.total_updates += 1
        if update_applied:
            self.successful_updates += 1
        
        self.kl_divergences.append(kl_div.item())
        
        # Clear experience buffers
        self.clear_experience()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_div.item(),
            'update_applied': update_applied,
            'total_loss': total_loss.item()
        }
    
    def clear_experience(self):
        """Clear experience buffers."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.entropies.clear()
    
    def act(self, state: np.ndarray) -> Tuple[int, float, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.local_network.get_action_and_value(state_tensor)
        
        return action.item(), log_prob.item(), entropy.item(), value.item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        log_prob: float, entropy: float, value: float):
        """Store experience in buffers."""
        self.states.append(torch.FloatTensor(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(torch.tensor(log_prob))
        self.entropies.append(torch.tensor(entropy))
        self.values.append(torch.tensor(value))
    
    def run_episode(self) -> Dict[str, float]:
        """
        Run one episode and collect experiences.
        
        Returns:
            episode_metrics: Dictionary of episode metrics
        """
        # Sync with global network
        self.sync_with_global()
        
        # Reset environment
        states = self.env.reset()
        my_state = states[self.worker_id % len(states)]  # Handle case where fewer agents than workers
        
        episode_reward = 0.0
        episode_length = 0
        step_count = 0
        
        while step_count < self.max_steps:
            # Select action
            action, log_prob, entropy, value = self.act(my_state)
            
            # Take action in environment (for multi-agent, we need to handle other agents)
            # For now, use random actions for other agents
            actions = [action if i == (self.worker_id % self.env.num_agents) 
                      else np.random.randint(0, 4) for i in range(self.env.num_agents)]
            
            next_states, rewards, dones, info = self.env.step(actions)
            my_reward = rewards[self.worker_id % len(rewards)]
            my_next_state = next_states[self.worker_id % len(next_states)]
            
            # Store experience
            self.store_experience(my_state, action, my_reward, log_prob, entropy, value)
            
            # Update state and tracking
            my_state = my_next_state
            episode_reward += my_reward
            episode_length += 1
            step_count += 1
            self.total_steps += 1
            
            # Check if episode is done
            if dones[self.worker_id % len(dones)]:
                break
        
        # Compute next state value for bootstrapping
        if step_count == self.max_steps:
            # Bootstrap from next state
            with torch.no_grad():
                next_value = self.local_network.get_value(
                    torch.FloatTensor(my_state).unsqueeze(0)
                ).item()
        else:
            # Episode terminated
            next_value = 0.0
        
        # Update global network
        update_metrics = self.update_global_network(next_value)
        
        # Track episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episodes_completed += 1
        
        # Combine metrics
        episode_metrics = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed,
            **update_metrics
        }
        
        return episode_metrics
    
    def start(self):
        """Start the worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.start()
    
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _run_loop(self):
        """Main worker loop that runs in separate thread."""
        while self.running:
            try:
                metrics = self.run_episode()
                # Could log metrics here or send to main thread
                time.sleep(0.01)  # Small delay to prevent overwhelming
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                break
    
    def get_statistics(self) -> Dict[str, float]:
        """Get worker performance statistics."""
        return {
            'worker_id': self.worker_id,
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'avg_kl_divergence': np.mean(self.kl_divergences) if self.kl_divergences else 0.0,
            'update_success_rate': self.successful_updates / max(1, self.total_updates),
            'successful_updates': self.successful_updates,
            'total_updates': self.total_updates
        }


class A3CManager:
    """
    Manager for multiple A3C workers with knowledge sharing.
    
    Coordinates multiple workers and handles:
    - Global network management
    - Knowledge sharing between agents
    - Performance monitoring
    - Model saving/loading
    """
    
    def __init__(self,
                 num_workers: int,
                 env_factory,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 trust_region_coef: float = 0.01,
                 sharing_interval: int = 1000):
        
        self.num_workers = num_workers
        self.env_factory = env_factory
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sharing_interval = sharing_interval
        
        # Create global network
        self.global_network = SharedGlobalNetwork(
            state_dim, action_dim, max_kl=trust_region_coef
        )
        
        # Create workers
        self.workers = []
        for i in range(num_workers):
            worker = A3CWorker(
                worker_id=i,
                global_network=self.global_network,
                env_factory=env_factory,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                trust_region_coef=trust_region_coef
            )
            self.workers.append(worker)
        
        # Training state
        self.training = False
        self.total_steps = 0
        self.last_sharing_step = 0
    
    def start_training(self):
        """Start all workers."""
        self.training = True
        for worker in self.workers:
            worker.start()
    
    def stop_training(self):
        """Stop all workers."""
        self.training = False
        for worker in self.workers:
            worker.stop()
    
    def share_knowledge(self):
        """
        Share knowledge between agents.
        
        This could involve:
        - Averaging network parameters
        - Sharing best experiences
        - Updating exploration strategies
        """
        # For now, knowledge sharing happens automatically through the global network
        # Could implement more sophisticated sharing mechanisms here
        self.last_sharing_step = self.total_steps
        
        # Log sharing event
        print(f"Knowledge sharing at step {self.total_steps}")
    
    def get_global_statistics(self) -> Dict[str, float]:
        """Get aggregated statistics from all workers."""
        worker_stats = [worker.get_statistics() for worker in self.workers]
        
        total_steps = sum(stats['total_steps'] for stats in worker_stats)
        total_episodes = sum(stats['episodes_completed'] for stats in worker_stats)
        avg_rewards = [stats['avg_episode_reward'] for stats in worker_stats if stats['avg_episode_reward'] > 0]
        avg_kl_divs = [stats['avg_kl_divergence'] for stats in worker_stats if stats['avg_kl_divergence'] > 0]
        
        return {
            'total_steps': total_steps,
            'total_episodes': total_episodes,
            'avg_reward_across_workers': np.mean(avg_rewards) if avg_rewards else 0.0,
            'avg_kl_divergence': np.mean(avg_kl_divs) if avg_kl_divs else 0.0,
            'num_active_workers': len([w for w in self.workers if w.running]),
            'global_network_updates': self.global_network.update_count
        }
    
    def save_model(self, filepath: str):
        """Save the global network."""
        torch.save({
            'global_network_state_dict': self.global_network.state_dict(),
            'training_stats': self.get_global_statistics()
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the global network."""
        checkpoint = torch.load(filepath)
        self.global_network.load_state_dict(checkpoint['global_network_state_dict'])
        
        # Sync all workers with loaded model
        for worker in self.workers:
            worker.sync_with_global()


if __name__ == "__main__":
    # Test the A3C implementation
    from ..environment.competitive_waterworld import CompetitiveWaterworld
    
    def env_factory():
        return CompetitiveWaterworld(num_agents=4)
    
    # Create test environment to get dimensions
    test_env = env_factory()
    state_dim, action_dim = test_env.get_state_action_dims()
    
    print("Testing A3C Agent Implementation")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create A3C manager
    manager = A3CManager(
        num_workers=2,
        env_factory=env_factory,
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    print(f"Created A3C manager with {len(manager.workers)} workers")
    
    # Test worker creation and basic functionality
    worker = manager.workers[0]
    test_state = np.random.randn(state_dim)
    action, log_prob, entropy, value = worker.act(test_state)
    
    print(f"Worker test successful!")
    print(f"Action: {action}, Value: {value:.3f}")
    print(f"Log prob: {log_prob:.3f}, Entropy: {entropy:.3f}")
    
    print("A3C implementation test completed!")

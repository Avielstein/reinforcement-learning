"""
TD learning algorithms implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from collections import deque

from ..config.td_config import TDConfig
from ..models.td_critic import TDCritic, DuelingTDCritic
from .replay_buffer import ReplayBuffer, Experience


class TDLearner:
    """Main TD learning algorithm implementation."""
    
    def __init__(self, obs_dim: int, config: TDConfig, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.config = config
        self.device = device
        
        # Create critic network based on configuration
        if config.use_dueling:
            self.critic = DuelingTDCritic(
                obs_dim, 
                config.value_network_hidden,
                use_target_network=config.use_target_network
            ).to(device)
        else:
            self.critic = TDCritic(
                obs_dim,
                config.value_network_hidden,
                use_target_network=config.use_target_network
            ).to(device)
        
        # Set TD parameters
        self.critic.set_td_params(config.lambda_param, config.gamma)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.value_lr)
        
        # Experience replay
        if config.use_replay:
            self.replay_buffer = ReplayBuffer(
                config.replay_buffer_size,
                config.prioritized_replay,
                config.priority_alpha
            )
        else:
            self.replay_buffer = None
        
        # Training state
        self.step_count = 0
        self.update_count = 0
        self.td_errors = deque(maxlen=1000)
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.lr_decay
        )
    
    def compute_td_error(self, obs: torch.Tensor, reward: torch.Tensor,
                        next_obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute TD error based on selected method."""
        if self.config.method == 'td_0':
            return self._compute_td_0_error(obs, reward, next_obs, done)
        elif self.config.method == 'td_lambda':
            return self._compute_td_lambda_error(obs, reward, next_obs, done)
        elif self.config.method == 'n_step_td':
            return self._compute_n_step_td_error(obs, reward, next_obs, done)
        else:
            raise ValueError(f"Unknown TD method: {self.config.method}")
    
    def _compute_td_0_error(self, obs: torch.Tensor, reward: torch.Tensor,
                           next_obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute standard TD(0) error."""
        return self.critic.compute_td_error(obs, reward, next_obs, done)
    
    def _compute_td_lambda_error(self, obs: torch.Tensor, reward: torch.Tensor,
                                next_obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute TD(λ) error with eligibility traces."""
        td_error = self.critic.compute_td_error(obs, reward, next_obs, done)
        
        # Update eligibility traces
        self.critic.update_eligibility_traces(obs, td_error, self.config.replace_traces)
        
        return td_error
    
    def _compute_n_step_td_error(self, obs: torch.Tensor, reward: torch.Tensor,
                                next_obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute n-step TD error."""
        return self.critic.compute_n_step_td_error(
            obs, reward, next_obs, done, self.config.n_steps
        )
    
    def update(self, obs: np.ndarray, action: np.ndarray, reward: float,
               next_obs: np.ndarray, done: bool) -> Dict[str, float]:
        """Update TD learner with single experience."""
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)
        
        # Compute TD error
        td_error = self.compute_td_error(obs_tensor, reward_tensor, next_obs_tensor, done_tensor)
        
        # Store experience in replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add(obs, action, reward, next_obs, done, td_error.item())
        
        # Update statistics
        self.td_errors.append(abs(td_error.item()))
        self.step_count += 1
        
        # Perform update
        if self.replay_buffer is not None and self.replay_buffer.is_ready(self.config.replay_start_size):
            return self._update_from_replay()
        else:
            return self._update_online(td_error)
    
    def _update_online(self, td_error: torch.Tensor) -> Dict[str, float]:
        """Perform online TD update."""
        # Compute loss
        loss = td_error.pow(2).mean()
        
        # Clip TD error if configured
        if self.config.clip_td_error:
            td_error = torch.clamp(td_error, -self.config.td_error_clip, self.config.td_error_clip)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip_norm)
        
        self.optimizer.step()
        
        # Update target network
        if self.config.use_target_network and self.step_count % self.config.target_update_frequency == 0:
            self.critic.update_target_network(self.config.soft_update_tau)
        
        # Decay eligibility traces
        if self.config.method == 'td_lambda':
            self.critic.decay_eligibility_traces()
        
        self.update_count += 1
        
        return {
            'td_error': abs(td_error.item()),
            'value_loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def _update_from_replay(self) -> Dict[str, float]:
        """Perform update using experience replay."""
        # Sample batch from replay buffer
        batch, indices, weights = self.replay_buffer.sample(
            self.config.batch_size, 
            self.config.priority_beta
        )
        
        # Convert batch to tensors
        obs_batch = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        action_batch = torch.FloatTensor([exp.action for exp in batch]).to(self.device)
        reward_batch = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_obs_batch = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        done_batch = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Compute TD errors
        td_errors = self.compute_td_error(obs_batch, reward_batch, next_obs_batch, done_batch)
        
        # Clip TD errors if configured
        if self.config.clip_td_error:
            td_errors = torch.clamp(td_errors, -self.config.td_error_clip, self.config.td_error_clip)
        
        # Compute weighted loss
        loss = (weights_tensor * td_errors.pow(2)).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip_norm)
        
        self.optimizer.step()
        
        # Update priorities in replay buffer
        if self.config.prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Update target network
        if self.config.use_target_network and self.step_count % self.config.target_update_frequency == 0:
            self.critic.update_target_network(self.config.soft_update_tau)
        
        self.update_count += 1
        
        return {
            'td_error': abs(td_errors.mean().item()),
            'value_loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'replay_buffer_size': len(self.replay_buffer) if self.replay_buffer else 0
        }
    
    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate for observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            value = self.critic(obs_tensor)
            return value.item()
    
    def reset_episode(self):
        """Reset for new episode."""
        if self.config.method == 'td_lambda':
            self.critic.reset_eligibility_traces()
    
    def decay_learning_rate(self):
        """Decay learning rate."""
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr > self.config.min_lr:
            self.lr_scheduler.step()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'step_count': self.step_count,
            'update_count': self.update_count,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'avg_td_error': np.mean(self.td_errors) if self.td_errors else 0.0,
            'std_td_error': np.std(self.td_errors) if self.td_errors else 0.0
        }
        
        # Add replay buffer stats
        if self.replay_buffer is not None:
            stats.update(self.replay_buffer.get_stats())
        
        # Add eligibility trace stats
        if self.config.method == 'td_lambda' and self.config.log_eligibility_traces:
            stats.update(self.critic.get_eligibility_trace_stats())
        
        return stats
    
    def save(self, filepath: str):
        """Save TD learner state."""
        state = {
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'config': self.config
        }
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """Load TD learner state."""
        state = torch.load(filepath, map_location=self.device)
        self.critic.load_state_dict(state['critic_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.step_count = state['step_count']
        self.update_count = state['update_count']


class MultiStepTDLearner(TDLearner):
    """TD learner with multi-step returns."""
    
    def __init__(self, obs_dim: int, config: TDConfig, device: str = 'cpu'):
        super().__init__(obs_dim, config, device)
        
        # Buffer for multi-step returns
        self.multi_step_buffer = deque(maxlen=config.n_steps)
    
    def update(self, obs: np.ndarray, action: np.ndarray, reward: float,
               next_obs: np.ndarray, done: bool) -> Dict[str, float]:
        """Update with multi-step returns."""
        # Add experience to multi-step buffer
        self.multi_step_buffer.append((obs, action, reward, next_obs, done))
        
        # Only update when buffer is full or episode ends
        if len(self.multi_step_buffer) == self.config.n_steps or done:
            return self._update_multi_step()
        
        return {}
    
    def _update_multi_step(self) -> Dict[str, float]:
        """Perform multi-step TD update."""
        if not self.multi_step_buffer:
            return {}
        
        # Calculate multi-step return
        multi_step_return = 0.0
        gamma_power = 1.0
        
        for i, (_, _, reward, _, episode_done) in enumerate(self.multi_step_buffer):
            multi_step_return += gamma_power * reward
            gamma_power *= self.config.gamma
            
            if episode_done:
                break
        
        # Get first and last states
        first_obs, first_action, _, _, _ = self.multi_step_buffer[0]
        last_obs, _, _, last_next_obs, last_done = self.multi_step_buffer[-1]
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(first_obs).unsqueeze(0).to(self.device)
        next_obs_tensor = torch.FloatTensor(last_next_obs).unsqueeze(0).to(self.device)
        return_tensor = torch.FloatTensor([multi_step_return]).to(self.device)
        done_tensor = torch.BoolTensor([last_done]).to(self.device)
        
        # Compute TD error
        current_value = self.critic(obs_tensor)
        
        with torch.no_grad():
            next_value = self.critic(next_obs_tensor, use_target=True)
            target_value = return_tensor + (gamma_power * next_value * (1 - done_tensor.float()))
        
        td_error = target_value - current_value
        
        # Update
        return self._update_online(td_error)


class EnsembleTDLearner:
    """Ensemble of TD learners for uncertainty estimation."""
    
    def __init__(self, obs_dim: int, config: TDConfig, num_learners: int = 3, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.config = config
        self.num_learners = num_learners
        self.device = device
        
        # Create ensemble of learners
        self.learners = [
            TDLearner(obs_dim, config, device) 
            for _ in range(num_learners)
        ]
        
        # Initialize with different random seeds
        for i, learner in enumerate(self.learners):
            torch.manual_seed(i)
            learner.critic.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with different random seeds."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def update(self, obs: np.ndarray, action: np.ndarray, reward: float,
               next_obs: np.ndarray, done: bool) -> Dict[str, float]:
        """Update all learners in ensemble."""
        all_stats = []
        
        for learner in self.learners:
            stats = learner.update(obs, action, reward, next_obs, done)
            all_stats.append(stats)
        
        # Aggregate statistics
        if all_stats and all_stats[0]:  # Check if stats are not empty
            aggregated_stats = {}
            for key in all_stats[0].keys():
                values = [stats[key] for stats in all_stats if key in stats]
                if values:
                    aggregated_stats[f'mean_{key}'] = np.mean(values)
                    aggregated_stats[f'std_{key}'] = np.std(values)
            
            return aggregated_stats
        
        return {}
    
    def get_value(self, obs: np.ndarray) -> Tuple[float, float]:
        """Get value estimate with uncertainty."""
        values = [learner.get_value(obs) for learner in self.learners]
        return np.mean(values), np.std(values)
    
    def reset_episode(self):
        """Reset all learners for new episode."""
        for learner in self.learners:
            learner.reset_episode()
    
    def save(self, filepath: str):
        """Save ensemble state."""
        states = [learner.critic.state_dict() for learner in self.learners]
        torch.save(states, filepath)
    
    def load(self, filepath: str):
        """Load ensemble state."""
        states = torch.load(filepath, map_location=self.device)
        for learner, state in zip(self.learners, states):
            learner.critic.load_state_dict(state)

"""
TD learning specific critic networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .networks import create_mlp, DuelingNetwork


class TDCritic(nn.Module):
    """TD learning critic with eligibility traces support."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [128, 128, 64],
                 activation: str = 'relu', use_target_network: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.use_target_network = use_target_network
        
        # Main value network
        self.value_net = create_mlp(obs_dim, hidden_dims, 1, activation=activation)
        
        # Target network for stable learning
        if use_target_network:
            self.target_net = create_mlp(obs_dim, hidden_dims, 1, activation=activation)
            self.target_net.load_state_dict(self.value_net.state_dict())
            
            # Freeze target network
            for param in self.target_net.parameters():
                param.requires_grad = False
        
        # Eligibility traces
        self.eligibility_traces = {}
        self.lambda_param = 0.9
        self.gamma = 0.99
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Forward pass through value network."""
        if use_target and self.use_target_network:
            return self.target_net(obs).squeeze(-1)
        else:
            return self.value_net(obs).squeeze(-1)
    
    def compute_td_error(self, obs: torch.Tensor, reward: torch.Tensor, 
                        next_obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute TD error for current transition."""
        current_value = self.forward(obs)
        
        with torch.no_grad():
            next_value = self.forward(next_obs, use_target=True)
            target_value = reward + self.gamma * next_value * (1 - done.float())
        
        td_error = target_value - current_value
        return td_error
    
    def compute_n_step_td_error(self, obs_batch: torch.Tensor, reward_batch: torch.Tensor,
                               next_obs_batch: torch.Tensor, done_batch: torch.Tensor,
                               n_steps: int = 5) -> torch.Tensor:
        """Compute n-step TD error."""
        batch_size = obs_batch.size(0)
        current_values = self.forward(obs_batch)
        
        with torch.no_grad():
            # Calculate n-step returns
            returns = torch.zeros_like(current_values)
            
            for i in range(batch_size):
                n_step_return = 0.0
                gamma_power = 1.0
                
                # Sum discounted rewards for n steps
                for step in range(min(n_steps, batch_size - i)):
                    if i + step < batch_size:
                        n_step_return += gamma_power * reward_batch[i + step]
                        gamma_power *= self.gamma
                        
                        if done_batch[i + step]:
                            break
                
                # Add discounted value of final state
                if i + n_steps < batch_size and not done_batch[i + n_steps - 1]:
                    final_value = self.forward(next_obs_batch[i + n_steps - 1], use_target=True)
                    n_step_return += gamma_power * final_value
                
                returns[i] = n_step_return
        
        td_error = returns - current_values
        return td_error
    
    def update_eligibility_traces(self, obs: torch.Tensor, td_error: torch.Tensor,
                                 replace_traces: bool = True):
        """Update eligibility traces for TD(λ)."""
        # Get gradients with respect to parameters
        value = self.forward(obs)
        
        # For batch processing, we need to sum the values to get a scalar
        if value.dim() > 0:
            value = value.sum()
        
        value.backward(retain_graph=True)
        
        # Update traces for each parameter
        for name, param in self.value_net.named_parameters():
            if param.grad is not None:
                if name not in self.eligibility_traces:
                    self.eligibility_traces[name] = torch.zeros_like(param)
                
                if replace_traces:
                    # Replace traces
                    self.eligibility_traces[name] = param.grad.clone()
                else:
                    # Accumulate traces
                    self.eligibility_traces[name] = (self.lambda_param * self.gamma * 
                                                   self.eligibility_traces[name] + param.grad)
    
    def apply_td_update(self, td_error: torch.Tensor, learning_rate: float):
        """Apply TD update using eligibility traces."""
        with torch.no_grad():
            for name, param in self.value_net.named_parameters():
                if name in self.eligibility_traces:
                    param.data += learning_rate * td_error.mean() * self.eligibility_traces[name]
    
    def decay_eligibility_traces(self):
        """Decay eligibility traces."""
        for name in self.eligibility_traces:
            self.eligibility_traces[name] *= self.lambda_param * self.gamma
    
    def reset_eligibility_traces(self):
        """Reset all eligibility traces to zero."""
        for name in self.eligibility_traces:
            self.eligibility_traces[name].zero_()
    
    def update_target_network(self, tau: float = 0.005):
        """Soft update of target network."""
        if not self.use_target_network:
            return
        
        with torch.no_grad():
            for target_param, main_param in zip(self.target_net.parameters(), 
                                              self.value_net.parameters()):
                target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)
    
    def hard_update_target_network(self):
        """Hard update of target network."""
        if not self.use_target_network:
            return
        
        self.target_net.load_state_dict(self.value_net.state_dict())
    
    def set_td_params(self, lambda_param: float, gamma: float):
        """Set TD learning parameters."""
        self.lambda_param = lambda_param
        self.gamma = gamma
    
    def get_eligibility_trace_stats(self) -> Dict[str, float]:
        """Get statistics about eligibility traces."""
        if not self.eligibility_traces:
            return {}
        
        stats = {}
        for name, trace in self.eligibility_traces.items():
            stats[f"{name}_mean"] = trace.mean().item()
            stats[f"{name}_std"] = trace.std().item()
            stats[f"{name}_max"] = trace.max().item()
            stats[f"{name}_min"] = trace.min().item()
        
        return stats


class DuelingTDCritic(nn.Module):
    """Dueling TD critic for improved value estimation."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu', use_target_network: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.use_target_network = use_target_network
        
        # Main dueling network
        self.dueling_net = DuelingNetwork(obs_dim, hidden_dims, activation)
        
        # Target network
        if use_target_network:
            self.target_dueling_net = DuelingNetwork(obs_dim, hidden_dims, activation)
            self.target_dueling_net.load_state_dict(self.dueling_net.state_dict())
            
            # Freeze target network
            for param in self.target_dueling_net.parameters():
                param.requires_grad = False
        
        # Eligibility traces
        self.eligibility_traces = {}
        self.lambda_param = 0.9
        self.gamma = 0.99
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Forward pass through dueling network."""
        if use_target and self.use_target_network:
            return self.target_dueling_net(obs)
        else:
            return self.dueling_net(obs)
    
    def compute_td_error(self, obs: torch.Tensor, reward: torch.Tensor,
                        next_obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute TD error using dueling architecture."""
        current_value = self.forward(obs)
        
        with torch.no_grad():
            next_value = self.forward(next_obs, use_target=True)
            target_value = reward + self.gamma * next_value * (1 - done.float())
        
        td_error = target_value - current_value
        return td_error
    
    def update_target_network(self, tau: float = 0.005):
        """Soft update of target network."""
        if not self.use_target_network:
            return
        
        with torch.no_grad():
            for target_param, main_param in zip(self.target_dueling_net.parameters(),
                                              self.dueling_net.parameters()):
                target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)
    
    def hard_update_target_network(self):
        """Hard update of target network."""
        if not self.use_target_network:
            return
        
        self.target_dueling_net.load_state_dict(self.dueling_net.state_dict())


class EnsembleTDCritic(nn.Module):
    """Ensemble of TD critics for uncertainty estimation."""
    
    def __init__(self, obs_dim: int, num_critics: int = 3, 
                 hidden_dims: List[int] = [128, 128, 64],
                 activation: str = 'relu', use_target_network: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.num_critics = num_critics
        self.use_target_network = use_target_network
        
        # Create ensemble of critics
        self.critics = nn.ModuleList([
            TDCritic(obs_dim, hidden_dims, activation, use_target_network)
            for _ in range(num_critics)
        ])
        
        # Initialize with different random seeds
        for i, critic in enumerate(self.critics):
            torch.manual_seed(i)
            critic.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, use_target: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and uncertainty."""
        values = torch.stack([critic(obs, use_target) for critic in self.critics])
        
        mean_value = values.mean(dim=0)
        uncertainty = values.std(dim=0)
        
        return mean_value, uncertainty
    
    def compute_td_error(self, obs: torch.Tensor, reward: torch.Tensor,
                        next_obs: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute TD error with uncertainty estimation."""
        td_errors = torch.stack([
            critic.compute_td_error(obs, reward, next_obs, done)
            for critic in self.critics
        ])
        
        mean_td_error = td_errors.mean(dim=0)
        td_uncertainty = td_errors.std(dim=0)
        
        return mean_td_error, td_uncertainty
    
    def update_target_networks(self, tau: float = 0.005):
        """Update all target networks."""
        for critic in self.critics:
            critic.update_target_network(tau)
    
    def set_td_params(self, lambda_param: float, gamma: float):
        """Set TD parameters for all critics."""
        for critic in self.critics:
            critic.set_td_params(lambda_param, gamma)
    
    def reset_eligibility_traces(self):
        """Reset eligibility traces for all critics."""
        for critic in self.critics:
            critic.reset_eligibility_traces()

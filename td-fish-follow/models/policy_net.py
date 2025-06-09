"""
Policy networks for continuous control in TD learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .networks import create_mlp


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous action spaces with exploration strategies."""
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu',
                 log_std_init: float = -0.5,
                 log_std_min: float = -20.0,
                 log_std_max: float = 2.0,
                 action_scale: float = 1.0):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = action_scale
        
        # Mean network
        self.mean_net = create_mlp(obs_dim, hidden_dims, action_dim, 
                                  activation=activation, output_activation='tanh')
        
        # Log standard deviation (can be learnable parameter or network output)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Action smoothing for temporal consistency
        self.action_smoothing = 0.7
        self.previous_action = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std."""
        mean = self.mean_net(obs) * self.action_scale
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False,
               with_log_prob: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample action from policy."""
        mean, std = self.forward(obs)
        
        if deterministic:
            action = mean
            log_prob = None if not with_log_prob else torch.zeros(obs.shape[0])
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            
            if with_log_prob:
                log_prob = normal.log_prob(action).sum(dim=-1)
            else:
                log_prob = None
        
        # Apply action smoothing if previous action exists
        if self.previous_action is not None and action.shape == self.previous_action.shape:
            action = (self.action_smoothing * self.previous_action + 
                     (1 - self.action_smoothing) * action)
        
        # Store for next smoothing
        self.previous_action = action.detach().clone()
        
        # Clamp actions to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob
    
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Calculate log probability of action."""
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        return normal.log_prob(action).sum(dim=-1)
    
    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Calculate policy entropy."""
        _, std = self.forward(obs)
        return torch.distributions.Normal(torch.zeros_like(std), std).entropy().sum(dim=-1)
    
    def reset_action_smoothing(self):
        """Reset action smoothing (call at episode start)."""
        self.previous_action = None
    
    def set_action_smoothing(self, smoothing: float):
        """Set action smoothing factor."""
        self.action_smoothing = smoothing


class AdaptivePolicyNetwork(nn.Module):
    """Policy network with adaptive exploration based on uncertainty."""
    
    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu',
                 uncertainty_threshold: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.uncertainty_threshold = uncertainty_threshold
        
        # Shared feature extractor
        self.feature_net = create_mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1], 
                                    activation=activation)
        
        # Mean and std networks
        self.mean_net = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_net = nn.Linear(hidden_dims[-1], action_dim)
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning mean, std, and uncertainty."""
        features = self.feature_net(obs)
        
        mean = torch.tanh(self.mean_net(features))
        log_std = torch.clamp(self.log_std_net(features), -20, 2)
        std = torch.exp(log_std)
        
        uncertainty = torch.sigmoid(self.uncertainty_net(features))
        
        return mean, std, uncertainty
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action with adaptive exploration."""
        mean, std, uncertainty = self.forward(obs)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0])
        else:
            # Increase exploration in uncertain states
            exploration_bonus = torch.where(
                uncertainty > self.uncertainty_threshold,
                uncertainty * 0.5,  # Increase std by up to 50%
                torch.zeros_like(uncertainty)
            )
            
            adjusted_std = std + exploration_bonus.unsqueeze(-1)
            normal = torch.distributions.Normal(mean, adjusted_std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1)
        
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob, uncertainty.squeeze(-1)


class HierarchicalPolicyNetwork(nn.Module):
    """Hierarchical policy for multi-level action selection."""
    
    def __init__(self, obs_dim: int, action_dim: int,
                 high_level_dim: int = 4,
                 hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu'):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.high_level_dim = high_level_dim
        
        # High-level policy (selects goals/strategies)
        self.high_level_policy = create_mlp(obs_dim, hidden_dims, high_level_dim,
                                          activation=activation, output_activation='softmax')
        
        # Low-level policies (one for each high-level action)
        self.low_level_policies = nn.ModuleList([
            ContinuousPolicyNetwork(obs_dim + high_level_dim, action_dim, hidden_dims, activation)
            for _ in range(high_level_dim)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through hierarchical policy."""
        # High-level policy
        high_level_probs = self.high_level_policy(obs)
        
        # Low-level policies
        low_level_outputs = []
        for policy in self.low_level_policies:
            # Concatenate observation with one-hot high-level action
            high_level_one_hot = torch.eye(self.high_level_dim).to(obs.device)
            extended_obs = torch.cat([obs.unsqueeze(1).repeat(1, self.high_level_dim, 1),
                                    high_level_one_hot.unsqueeze(0).repeat(obs.shape[0], 1, 1)], dim=-1)
            
            mean, std = policy(extended_obs.view(-1, extended_obs.shape[-1]))
            mean = mean.view(obs.shape[0], self.high_level_dim, -1)
            std = std.view(obs.shape[0], self.high_level_dim, -1)
            
            low_level_outputs.append((mean, std))
        
        return high_level_probs, low_level_outputs
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample hierarchical action."""
        high_level_probs, low_level_outputs = self.forward(obs)
        
        if deterministic:
            high_level_action = torch.argmax(high_level_probs, dim=-1)
        else:
            high_level_dist = torch.distributions.Categorical(high_level_probs)
            high_level_action = high_level_dist.sample()
        
        # Select corresponding low-level action
        batch_size = obs.shape[0]
        final_actions = torch.zeros(batch_size, self.action_dim).to(obs.device)
        log_probs = torch.zeros(batch_size).to(obs.device)
        
        for i in range(batch_size):
            hl_action = high_level_action[i].item()
            mean, std = low_level_outputs[hl_action]
            
            if deterministic:
                action = mean[i, hl_action]
                log_prob = torch.tensor(0.0)
            else:
                normal = torch.distributions.Normal(mean[i, hl_action], std[i, hl_action])
                action = normal.sample()
                log_prob = normal.log_prob(action).sum()
            
            final_actions[i] = action
            log_probs[i] = log_prob
        
        return final_actions, log_probs, high_level_action


class CuriositydrivenPolicyNetwork(nn.Module):
    """Policy network with intrinsic curiosity module."""
    
    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu',
                 curiosity_weight: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.curiosity_weight = curiosity_weight
        
        # Main policy network
        self.policy = ContinuousPolicyNetwork(obs_dim, action_dim, hidden_dims, activation)
        
        # Curiosity module (predicts next state)
        self.forward_model = create_mlp(obs_dim + action_dim, hidden_dims, obs_dim, activation)
        self.inverse_model = create_mlp(obs_dim * 2, hidden_dims, action_dim, activation)
        
        # Feature encoder for curiosity
        self.feature_encoder = create_mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1], activation)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through main policy."""
        return self.policy(obs)
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        return self.policy.sample(obs, deterministic)
    
    def compute_curiosity_reward(self, obs: torch.Tensor, action: torch.Tensor, 
                                next_obs: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic curiosity reward."""
        # Encode observations
        obs_features = self.feature_encoder(obs)
        next_obs_features = self.feature_encoder(next_obs)
        
        # Forward model prediction
        obs_action = torch.cat([obs_features, action], dim=-1)
        predicted_next_features = self.forward_model(obs_action)
        
        # Prediction error as curiosity reward
        prediction_error = F.mse_loss(predicted_next_features, next_obs_features, reduction='none')
        curiosity_reward = prediction_error.mean(dim=-1) * self.curiosity_weight
        
        return curiosity_reward
    
    def compute_inverse_loss(self, obs: torch.Tensor, next_obs: torch.Tensor, 
                           action: torch.Tensor) -> torch.Tensor:
        """Compute inverse model loss for training."""
        obs_features = self.feature_encoder(obs)
        next_obs_features = self.feature_encoder(next_obs)
        
        obs_next_obs = torch.cat([obs_features, next_obs_features], dim=-1)
        predicted_action = self.inverse_model(obs_next_obs)
        
        inverse_loss = F.mse_loss(predicted_action, action)
        return inverse_loss

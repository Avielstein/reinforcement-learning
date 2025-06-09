"""
Basic neural network architectures for TD learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def create_mlp(input_dim: int, hidden_dims: List[int], output_dim: int, 
               activation: str = 'relu', output_activation: Optional[str] = None) -> nn.Module:
    """Create a multi-layer perceptron."""
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    
    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'elu':
            layers.append(nn.ELU())
        
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
    
    # Final activation for hidden layers
    if activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'elu':
        layers.append(nn.ELU())
    
    # Output layer
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    
    # Output activation
    if output_activation == 'tanh':
        layers.append(nn.Tanh())
    elif output_activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif output_activation == 'softmax':
        layers.append(nn.Softmax(dim=-1))
    
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """Policy network for continuous action spaces."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu', log_std_init: float = -0.5):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Mean network
        self.mean_net = create_mlp(obs_dim, hidden_dims, action_dim, 
                                  activation=activation, output_activation='tanh')
        
        # Log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std."""
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
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


class ValueNetwork(nn.Module):
    """Value function network."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [128, 128, 64],
                 activation: str = 'relu'):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.network = create_mlp(obs_dim, hidden_dims, 1, activation=activation)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate."""
        return self.network(obs).squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """Combined actor-critic network."""
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 policy_hidden: List[int] = [128, 128],
                 value_hidden: List[int] = [128, 128, 64],
                 shared_layers: Optional[List[int]] = None,
                 activation: str = 'relu'):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared feature extractor (optional)
        if shared_layers:
            self.shared_net = create_mlp(obs_dim, shared_layers, shared_layers[-1], 
                                       activation=activation)
            feature_dim = shared_layers[-1]
        else:
            self.shared_net = None
            feature_dim = obs_dim
        
        # Policy network
        self.policy = PolicyNetwork(feature_dim, action_dim, policy_hidden, activation)
        
        # Value network
        self.value = ValueNetwork(feature_dim, value_hidden, activation)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean, std, and value."""
        if self.shared_net:
            features = self.shared_net(obs)
        else:
            features = obs
        
        mean, std = self.policy(features)
        value = self.value(features)
        
        return mean, std, value
    
    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return action, log_prob, value."""
        if self.shared_net:
            features = self.shared_net(obs)
        else:
            features = obs
        
        action, log_prob = self.policy.sample(features)
        value = self.value(features)
        
        return action, log_prob, value
    
    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action and return log_prob, value, entropy."""
        if self.shared_net:
            features = self.shared_net(obs)
        else:
            features = obs
        
        log_prob = self.policy.log_prob(features, action)
        value = self.value(features)
        entropy = self.policy.entropy(features)
        
        return log_prob, value, entropy


class DuelingNetwork(nn.Module):
    """Dueling network architecture for value function."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu'):
        super().__init__()
        
        self.obs_dim = obs_dim
        
        # Shared feature extractor
        self.feature_net = create_mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1], 
                                    activation=activation)
        
        # Value stream
        self.value_stream = nn.Linear(hidden_dims[-1], 1)
        
        # Advantage stream (for action-dependent values)
        self.advantage_stream = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture."""
        features = self.feature_net(obs)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_value.squeeze(-1)


class NoiseNetwork(nn.Module):
    """Noisy network for exploration."""
    
    def __init__(self, input_dim: int, output_dim: int, std_init: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.empty(output_dim))
        self.bias_sigma = nn.Parameter(torch.empty(output_dim))
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.empty(output_dim, input_dim))
        self.register_buffer('bias_epsilon', torch.empty(output_dim))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.output_dim))
    
    def reset_noise(self):
        """Reset noise buffers."""
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)

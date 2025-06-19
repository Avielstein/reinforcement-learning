"""Neural network architectures for RAINBOW DQN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not parameters)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise buffers."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int):
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input):
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)

class QNetwork(nn.Module):
    """Q-Network for Double DQN."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state space (sensor readings)
            action_dim: Dimension of action space (movement directions)
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through network."""
        return self.network(state)
    
    def get_action(self, state, epsilon=0.0):
        """Get action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            # Random action
            return np.random.randint(0, self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state)
                return q_values.argmax().item()


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        """
        Initialize Dueling Q-Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(DuelingQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        feature_layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through dueling network."""
        features = self.feature_layers(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """Get action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            # Random action
            return np.random.randint(0, self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state)
                return q_values.argmax().item()


class RainbowNetwork(nn.Module):
    """RAINBOW network combining Dueling + Noisy + Distributional RL."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: list = [512, 512],
        n_atoms: int = 51,
        noisy_std: float = 0.5
    ):
        """
        Initialize RAINBOW network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            n_atoms: Number of atoms for distributional RL
            noisy_std: Standard deviation for noisy layers
        """
        super(RainbowNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        
        # Shared feature layers (regular layers)
        feature_layers = []
        input_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            feature_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*feature_layers)
        
        # Noisy layers for value and advantage streams
        final_hidden = hidden_dims[-1]
        
        # Value stream (noisy)
        self.value_hidden = NoisyLinear(input_dim, final_hidden, noisy_std)
        self.value_out = NoisyLinear(final_hidden, n_atoms, noisy_std)
        
        # Advantage stream (noisy)
        self.advantage_hidden = NoisyLinear(input_dim, final_hidden, noisy_std)
        self.advantage_out = NoisyLinear(final_hidden, action_dim * n_atoms, noisy_std)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through RAINBOW network.
        
        Returns:
            Distribution over Q-values for each action
            Shape: (batch_size, action_dim, n_atoms)
        """
        batch_size = state.size(0)
        
        # Shared features
        features = self.feature_layers(state)
        
        # Value stream
        value_hidden = F.relu(self.value_hidden(features))
        value = self.value_out(value_hidden)  # (batch_size, n_atoms)
        
        # Advantage stream
        advantage_hidden = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_out(advantage_hidden)  # (batch_size, action_dim * n_atoms)
        advantage = advantage.view(batch_size, self.action_dim, self.n_atoms)
        
        # Combine value and advantage (dueling)
        value = value.unsqueeze(1).expand_as(advantage)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distributions
        q_dist = F.softmax(q_atoms, dim=2)
        
        return q_dist
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()

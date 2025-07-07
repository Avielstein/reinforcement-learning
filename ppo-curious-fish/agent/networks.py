import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    """PPO Actor network for continuous action space."""
    
    def __init__(self, state_dim=152, action_dim=4, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log std for continuous actions
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = torch.tanh(self.mean_head(x))  # Actions between -1 and 1
        log_std = torch.clamp(self.log_std_head(x), -20, 2)  # Clamp for stability
        
        return mean, log_std
    
    def get_action_and_log_prob(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample action from normal distribution
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def get_log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return log_prob

class CriticNetwork(nn.Module):
    """PPO Critic network for value estimation."""
    
    def __init__(self, state_dim=152, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_head(x)
        
        return value

class ICMNetwork(nn.Module):
    """Intrinsic Curiosity Module (ICM) network."""
    
    def __init__(self, state_dim=152, action_dim=4, feature_dim=64, hidden_dim=256):
        super(ICMNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Inverse model: predicts action from state features
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Forward model: predicts next state features from current features and action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0.0)
    
    def encode_state(self, state):
        """Encode state to feature representation."""
        return self.feature_encoder(state)
    
    def forward(self, state, action, next_state):
        """
        Forward pass through ICM.
        
        Returns:
            inverse_pred: predicted action
            forward_pred: predicted next state features
            state_features: current state features
            next_state_features: next state features
        """
        # Encode states to features
        state_features = self.encode_state(state)
        next_state_features = self.encode_state(next_state)
        
        # Inverse model: predict action from state transition
        inverse_input = torch.cat([state_features, next_state_features], dim=-1)
        inverse_pred = self.inverse_model(inverse_input)
        
        # Forward model: predict next state features from current features and action
        forward_input = torch.cat([state_features, action], dim=-1)
        forward_pred = self.forward_model(forward_input)
        
        return inverse_pred, forward_pred, state_features, next_state_features
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward based on prediction error."""
        with torch.no_grad():
            _, forward_pred, _, next_state_features = self.forward(state, action, next_state)
            
            # Intrinsic reward is the prediction error
            prediction_error = F.mse_loss(forward_pred, next_state_features, reduction='none')
            intrinsic_reward = prediction_error.mean(dim=-1)
            
        return intrinsic_reward

class FeatureEncoder(nn.Module):
    """Standalone feature encoder for curiosity visualization."""
    
    def __init__(self, state_dim=152, feature_dim=64, hidden_dim=256):
        super(FeatureEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()  # Normalize features
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        return self.encoder(state)

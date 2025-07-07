import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .networks import ICMNetwork

class IntrinsicCuriosityModule:
    """
    Intrinsic Curiosity Module (ICM) for PPO agent.
    
    Implements curiosity-driven exploration through prediction error.
    Based on "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)
    """
    
    def __init__(self, state_dim=152, action_dim=4, feature_dim=64, hidden_dim=256, 
                 learning_rate=1e-3, forward_loss_weight=0.2, inverse_loss_weight=0.8,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.forward_loss_weight = forward_loss_weight
        self.inverse_loss_weight = inverse_loss_weight
        self.device = device
        
        # Create ICM network
        self.icm_network = ICMNetwork(state_dim, action_dim, feature_dim, hidden_dim).to(device)
        
        # Optimizer for ICM
        self.optimizer = torch.optim.Adam(self.icm_network.parameters(), lr=learning_rate)
        
        # Track curiosity statistics
        self.curiosity_history = []
        self.prediction_errors = []
        
    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Compute intrinsic reward based on prediction error.
        
        Args:
            state: Current state tensor
            action: Action tensor
            next_state: Next state tensor
            
        Returns:
            intrinsic_reward: Curiosity-based reward
        """
        return self.icm_network.compute_intrinsic_reward(state, action, next_state)
    
    def update(self, states, actions, next_states):
        """
        Update ICM networks using collected experience.
        
        Args:
            states: Batch of current states
            actions: Batch of actions
            next_states: Batch of next states
            
        Returns:
            Dictionary with loss information
        """
        # Convert to tensors if needed
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions).to(self.device)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.FloatTensor(next_states).to(self.device)
        
        # Forward pass through ICM
        inverse_pred, forward_pred, state_features, next_state_features = \
            self.icm_network(states, actions, next_states)
        
        # Compute losses
        # Inverse loss: how well can we predict the action from state transition?
        inverse_loss = F.mse_loss(inverse_pred, actions)
        
        # Forward loss: how well can we predict the next state features?
        forward_loss = F.mse_loss(forward_pred, next_state_features.detach())
        
        # Total ICM loss
        icm_loss = self.inverse_loss_weight * inverse_loss + \
                   self.forward_loss_weight * forward_loss
        
        # Update ICM
        self.optimizer.zero_grad()
        icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm_network.parameters(), 0.5)
        self.optimizer.step()
        
        # Track statistics
        with torch.no_grad():
            prediction_error = F.mse_loss(forward_pred, next_state_features, reduction='none').mean(dim=-1)
            self.prediction_errors.extend(prediction_error.cpu().numpy().tolist())
            self.curiosity_history.append(prediction_error.mean().item())
        
        return {
            'icm_loss': icm_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'forward_loss': forward_loss.item(),
            'prediction_error': prediction_error.mean().item()
        }
    
    def get_curiosity_map(self, states, grid_size=20):
        """
        Generate a curiosity heat map for visualization.
        
        Args:
            states: Current states to evaluate
            grid_size: Size of the visualization grid
            
        Returns:
            Curiosity heat map as numpy array
        """
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            
            # Encode states to features
            features = self.icm_network.encode_state(states)
            
            # Create a simple curiosity map based on feature variance
            # Higher variance = more novel/curious areas
            feature_var = torch.var(features, dim=0)
            curiosity_score = feature_var.mean().item()
            
            # Create a simple heat map (this is a simplified version)
            heat_map = np.ones((grid_size, grid_size)) * curiosity_score
            
            return heat_map
    
    def get_feature_representation(self, state):
        """Get learned feature representation of a state."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            features = self.icm_network.encode_state(state)
            return features.cpu().numpy()
    
    def get_statistics(self):
        """Get curiosity module statistics."""
        if not self.curiosity_history:
            return {
                'avg_curiosity': 0.0,
                'recent_curiosity': 0.0,
                'curiosity_trend': 0.0,
                'total_updates': 0
            }
        
        recent_window = min(100, len(self.curiosity_history))
        recent_curiosity = np.mean(self.curiosity_history[-recent_window:])
        
        # Calculate trend (positive = increasing curiosity)
        if len(self.curiosity_history) > 10:
            early_curiosity = np.mean(self.curiosity_history[:10])
            trend = recent_curiosity - early_curiosity
        else:
            trend = 0.0
        
        return {
            'avg_curiosity': np.mean(self.curiosity_history),
            'recent_curiosity': recent_curiosity,
            'curiosity_trend': trend,
            'total_updates': len(self.curiosity_history),
            'max_curiosity': np.max(self.curiosity_history) if self.curiosity_history else 0.0,
            'min_curiosity': np.min(self.curiosity_history) if self.curiosity_history else 0.0
        }
    
    def reset_statistics(self):
        """Reset curiosity statistics."""
        self.curiosity_history.clear()
        self.prediction_errors.clear()
    
    def save(self, filepath):
        """Save ICM model."""
        torch.save({
            'icm_state_dict': self.icm_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'curiosity_history': self.curiosity_history,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'feature_dim': self.feature_dim,
                'forward_loss_weight': self.forward_loss_weight,
                'inverse_loss_weight': self.inverse_loss_weight
            }
        }, filepath)
    
    def load(self, filepath):
        """Load ICM model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.icm_network.load_state_dict(checkpoint['icm_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curiosity_history = checkpoint.get('curiosity_history', [])
        
        # Load config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.forward_loss_weight = config.get('forward_loss_weight', self.forward_loss_weight)
            self.inverse_loss_weight = config.get('inverse_loss_weight', self.inverse_loss_weight)

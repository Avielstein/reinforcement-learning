"""
Neural network policy for survival agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class PolicyNetwork(nn.Module):
    """
    Simple neural network that takes observations and outputs actions
    Designed to be fast and suitable for genetic evolution
    """
    
    def __init__(self, input_size: int = 53, hidden_size: int = 64, output_size: int = 4):
        """
        Initialize policy network
        
        Args:
            input_size: Size of observation vector (3 base + 10*5 nearby agents = 53)
            hidden_size: Size of hidden layers
            output_size: Size of action vector (move_x, move_y, speed, attack)
        """
        super(PolicyNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Simple 3-layer network for fast evolution
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights with small random values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small random values"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.zeros_(layer.bias)
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            observation: Input observation tensor
            
        Returns:
            Action tensor [move_x, move_y, speed, attack_probability]
        """
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        # Apply appropriate activations
        move_x = torch.tanh(output[0])  # Movement in x direction [-1, 1]
        move_y = torch.tanh(output[1])  # Movement in y direction [-1, 1]
        speed = torch.sigmoid(output[2])  # Speed multiplier [0, 1]
        attack = torch.sigmoid(output[3])  # Attack probability [0, 1]
        
        return torch.tensor([move_x, move_y, speed, attack])
    
    def get_action(self, observation: np.ndarray, add_noise: bool = False, noise_scale: float = 0.1) -> np.ndarray:
        """
        Get action from observation (convenience method)
        
        Args:
            observation: Numpy observation array
            add_noise: Whether to add exploration noise
            noise_scale: Scale of exploration noise
            
        Returns:
            Action as numpy array
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation)
            action = self.forward(obs_tensor)
            
            if add_noise:
                noise = torch.normal(0, noise_scale, size=action.shape)
                action = action + noise
                # Clamp to valid ranges
                action[0] = torch.clamp(action[0], -1, 1)  # move_x
                action[1] = torch.clamp(action[1], -1, 1)  # move_y
                action[2] = torch.clamp(action[2], 0, 1)   # speed
                action[3] = torch.clamp(action[3], 0, 1)   # attack
            
            return action.numpy()
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.05):
        """
        Apply random mutations to network weights (for genetic evolution)
        
        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation of mutation noise
        """
        with torch.no_grad():
            for param in self.parameters():
                # Create mutation mask
                mutation_mask = torch.rand_like(param) < mutation_rate
                
                # Apply mutations
                mutations = torch.normal(0, mutation_strength, size=param.shape)
                param.data += mutation_mask.float() * mutations
    
    def crossover(self, other_network: 'PolicyNetwork', crossover_rate: float = 0.5) -> 'PolicyNetwork':
        """
        Create offspring network by crossing over with another network
        
        Args:
            other_network: Other parent network
            crossover_rate: Probability of taking weights from this network vs other
            
        Returns:
            New offspring network
        """
        offspring = PolicyNetwork(self.input_size, self.hidden_size, self.output_size)
        
        with torch.no_grad():
            for (name, param), (_, other_param), (_, offspring_param) in zip(
                self.named_parameters(), 
                other_network.named_parameters(), 
                offspring.named_parameters()
            ):
                # Create crossover mask
                crossover_mask = torch.rand_like(param) < crossover_rate
                
                # Combine parameters
                offspring_param.data = torch.where(
                    crossover_mask, 
                    param.data, 
                    other_param.data
                )
        
        return offspring
    
    def copy(self) -> 'PolicyNetwork':
        """Create a deep copy of this network"""
        copy_network = PolicyNetwork(self.input_size, self.hidden_size, self.output_size)
        copy_network.load_state_dict(self.state_dict())
        return copy_network
    
    def get_weights_vector(self) -> np.ndarray:
        """Get all network weights as a single vector (for analysis)"""
        weights = []
        with torch.no_grad():
            for param in self.parameters():
                weights.append(param.data.flatten())
        return torch.cat(weights).numpy()
    
    def set_weights_vector(self, weights_vector: np.ndarray):
        """Set network weights from a vector"""
        weights_tensor = torch.FloatTensor(weights_vector)
        start_idx = 0
        
        with torch.no_grad():
            for param in self.parameters():
                param_size = param.numel()
                param_weights = weights_tensor[start_idx:start_idx + param_size]
                param.data = param_weights.view(param.shape)
                start_idx += param_size
    
    def calculate_similarity(self, other_network: 'PolicyNetwork') -> float:
        """
        Calculate similarity between this network and another
        
        Args:
            other_network: Network to compare with
            
        Returns:
            Similarity score between 0 and 1
        """
        weights1 = self.get_weights_vector()
        weights2 = other_network.get_weights_vector()
        
        # Calculate cosine similarity
        dot_product = np.dot(weights1, weights2)
        norm1 = np.linalg.norm(weights1)
        norm2 = np.linalg.norm(weights2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return (similarity + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    def save(self, filepath: str):
        """Save network to file"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load network from file"""
        self.load_state_dict(torch.load(filepath))
    
    def get_network_info(self) -> dict:
        """Get information about the network structure"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': 3
        }

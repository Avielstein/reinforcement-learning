"""
Neural network policies that evolve through genetic algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

class PolicyNetwork(nn.Module):
    """Evolving neural network for tactical decision making"""
    
    def __init__(self, input_size: int = 12, hidden_size: int = 64, output_size: int = 6):
        super().__init__()
        
        # Network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Multiple output heads for different decisions
        self.movement_head = nn.Linear(hidden_size // 2, 2)  # x, y movement
        self.combat_head = nn.Linear(hidden_size // 2, 2)    # fire decision, target selection
        self.tactical_head = nn.Linear(hidden_size // 2, 2)  # cooperation, exploration
        
        # Initialize weights with small random values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the network"""
        # Shared layers
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        
        # Specialized output heads
        movement = torch.tanh(self.movement_head(h3))  # [-1, 1] for movement
        combat = torch.sigmoid(self.combat_head(h3))   # [0, 1] for combat decisions
        tactical = torch.sigmoid(self.tactical_head(h3))  # [0, 1] for tactical decisions
        
        return {
            'movement': movement,
            'combat': combat,
            'tactical': tactical
        }
    
    def get_action(self, observation: np.ndarray) -> Dict[str, float]:
        """Get action from observation"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            outputs = self.forward(obs_tensor)
            
            # Extract actions
            movement = outputs['movement'].squeeze().numpy()
            combat = outputs['combat'].squeeze().numpy()
            tactical = outputs['tactical'].squeeze().numpy()
            
            return {
                'move_x': float(movement[0]),
                'move_y': float(movement[1]),
                'should_fire': float(combat[0]),
                'target_preference': float(combat[1]),  # 0=weak targets, 1=strong targets
                'cooperation': float(tactical[0]),
                'exploration': float(tactical[1])
            }
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> 'PolicyNetwork':
        """Create a mutated copy of this network"""
        # Create a copy
        mutated = PolicyNetwork(
            input_size=self.fc1.in_features,
            hidden_size=self.fc1.out_features,
            output_size=6
        )
        mutated.load_state_dict(self.state_dict())
        
        # Mutate weights
        with torch.no_grad():
            for param in mutated.parameters():
                if random.random() < mutation_rate:
                    # Add Gaussian noise to weights
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)
        
        return mutated
    
    def crossover(self, other: 'PolicyNetwork', crossover_rate: float = 0.5) -> 'PolicyNetwork':
        """Create offspring by crossing over with another network"""
        offspring = PolicyNetwork(
            input_size=self.fc1.in_features,
            hidden_size=self.fc1.out_features,
            output_size=6
        )
        
        # Crossover weights layer by layer
        with torch.no_grad():
            self_params = list(self.parameters())
            other_params = list(other.parameters())
            offspring_params = list(offspring.parameters())
            
            for i, (self_param, other_param, offspring_param) in enumerate(
                zip(self_params, other_params, offspring_params)
            ):
                if random.random() < crossover_rate:
                    # Take from other parent
                    offspring_param.copy_(other_param)
                else:
                    # Take from this parent
                    offspring_param.copy_(self_param)
        
        return offspring
    
    def get_weight_diversity(self, other: 'PolicyNetwork') -> float:
        """Calculate diversity between two networks"""
        total_diff = 0.0
        total_params = 0
        
        with torch.no_grad():
            self_params = list(self.parameters())
            other_params = list(other.parameters())
            
            for self_param, other_param in zip(self_params, other_params):
                diff = torch.abs(self_param - other_param).mean().item()
                total_diff += diff
                total_params += 1
        
        return total_diff / total_params if total_params > 0 else 0.0
    
    def get_complexity_score(self) -> float:
        """Calculate network complexity (for analysis)"""
        total_weights = 0
        active_weights = 0
        
        with torch.no_grad():
            for param in self.parameters():
                total_weights += param.numel()
                active_weights += (torch.abs(param) > 0.01).sum().item()
        
        return active_weights / total_weights if total_weights > 0 else 0.0

class SpecializedPolicyNetwork(PolicyNetwork):
    """Policy network with specialized architectures for different strategies"""
    
    def __init__(self, strategy: str = 'balanced', **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self._specialize_for_strategy()
    
    def _specialize_for_strategy(self):
        """Modify network architecture based on strategy"""
        if self.strategy == 'aggressive':
            # Bias combat head towards aggression
            with torch.no_grad():
                self.combat_head.bias[0] += 0.5  # More likely to fire
                self.tactical_head.bias[1] -= 0.3  # Less exploration
        
        elif self.strategy == 'defensive':
            # Bias towards defensive behavior
            with torch.no_grad():
                self.combat_head.bias[0] -= 0.3  # Less likely to fire
                self.tactical_head.bias[0] += 0.4  # More cooperation
        
        elif self.strategy == 'scout':
            # Bias towards exploration and mobility
            with torch.no_grad():
                self.tactical_head.bias[1] += 0.5  # More exploration
                self.movement_head.weight *= 1.2  # More responsive movement
        
        elif self.strategy == 'sniper':
            # Bias towards precision and patience
            with torch.no_grad():
                self.combat_head.bias[1] += 0.4  # Prefer strong targets
                self.movement_head.weight *= 0.8  # Less movement

class EnsemblePolicyNetwork(nn.Module):
    """Ensemble of multiple policy networks for robust decision making"""
    
    def __init__(self, num_networks: int = 3, **kwargs):
        super().__init__()
        self.networks = nn.ModuleList([
            PolicyNetwork(**kwargs) for _ in range(num_networks)
        ])
        self.num_networks = num_networks
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        outputs = [net(x) for net in self.networks]
        
        # Average outputs
        ensemble_output = {}
        for key in outputs[0].keys():
            ensemble_output[key] = torch.mean(
                torch.stack([out[key] for out in outputs]), dim=0
            )
        
        return ensemble_output
    
    def get_action(self, observation: np.ndarray) -> Dict[str, float]:
        """Get action from ensemble"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            outputs = self.forward(obs_tensor)
            
            movement = outputs['movement'].squeeze().numpy()
            combat = outputs['combat'].squeeze().numpy()
            tactical = outputs['tactical'].squeeze().numpy()
            
            return {
                'move_x': float(movement[0]),
                'move_y': float(movement[1]),
                'should_fire': float(combat[0]),
                'target_preference': float(combat[1]),
                'cooperation': float(tactical[0]),
                'exploration': float(tactical[1])
            }
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> 'EnsemblePolicyNetwork':
        """Mutate the ensemble"""
        mutated = EnsemblePolicyNetwork(self.num_networks)
        
        for i, network in enumerate(self.networks):
            mutated.networks[i] = network.mutate(mutation_rate, mutation_strength)
        
        return mutated

"""
Genetic evolution manager for neural network policies
"""

import numpy as np
import torch
import random
from typing import Dict, List, Any
from ..core.config import Config

class GeneticEvolutionManager:
    """Manages genetic evolution of neural network policies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mutation_rate = config.mutation_rate
        self.crossover_rate = config.crossover_rate
        self.elite_ratio = config.elite_ratio
        
    def evolve_population(self, agents: List, fitness_scores: Dict[str, float]) -> None:
        """Evolve population based on fitness scores"""
        # This is handled in the main experiment class
        # This class provides utility methods for genetic operations
        pass
    
    def select_parents(self, agents: List, fitness_scores: List[float]) -> tuple:
        """Select two parents using tournament selection"""
        tournament_size = 3
        
        # Tournament selection for parent 1
        tournament1 = random.sample(list(zip(agents, fitness_scores)), min(tournament_size, len(agents)))
        parent1 = max(tournament1, key=lambda x: x[1])[0]
        
        # Tournament selection for parent 2
        tournament2 = random.sample(list(zip(agents, fitness_scores)), min(tournament_size, len(agents)))
        parent2 = max(tournament2, key=lambda x: x[1])[0]
        
        return parent1, parent2
    
    def crossover_networks(self, network1, network2, crossover_rate: float = None) -> tuple:
        """Perform crossover between two neural networks"""
        if crossover_rate is None:
            crossover_rate = self.crossover_rate
        
        offspring1 = network1.crossover(network2, crossover_rate)
        offspring2 = network2.crossover(network1, crossover_rate)
        
        return offspring1, offspring2
    
    def mutate_network(self, network, mutation_rate: float = None, mutation_strength: float = 0.2):
        """Mutate a neural network"""
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        
        return network.mutate(mutation_rate, mutation_strength)
    
    def calculate_diversity(self, agents: List) -> float:
        """Calculate genetic diversity within a population"""
        if len(agents) < 2:
            return 0.0
        
        total_diversity = 0.0
        comparisons = 0
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                diversity = agents[i].policy.get_weight_diversity(agents[j].policy)
                total_diversity += diversity
                comparisons += 1
        
        return total_diversity / max(comparisons, 1)

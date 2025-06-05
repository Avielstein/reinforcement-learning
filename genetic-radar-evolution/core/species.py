"""
Species class for genetic evolution
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Species:
    """Represents a species with evolving neural network policies"""
    
    id: str
    name: str
    color: str
    population_size: int
    generation: int = 0
    
    # Evolution tracking
    fitness_history: List[float] = field(default_factory=list)
    best_fitness: float = -float('inf')
    
    # Performance metrics
    total_kills: int = 0
    total_deaths: int = 0
    total_damage_dealt: float = 0.0
    battles_won: int = 0
    battles_participated: int = 0
    
    def update_fitness(self, fitness: float) -> None:
        """Update fitness history"""
        self.fitness_history.append(fitness)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
    
    def get_average_fitness(self, last_n: int = 5) -> float:
        """Get average fitness over last N generations"""
        if not self.fitness_history:
            return 0.0
        recent = self.fitness_history[-last_n:]
        return np.mean(recent)

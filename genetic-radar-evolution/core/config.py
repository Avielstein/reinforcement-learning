"""
Configuration settings for genetic radar evolution
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    """Global configuration for the genetic evolution system"""
    
    # Environment settings
    map_size: float = 1000.0
    max_simulation_time: float = 300.0  # 5 minutes
    dt: float = 0.1  # Time step
    
    # Combat settings
    projectile_speed: float = 300.0
    max_radar_range: float = 300.0
    max_firing_range: float = 200.0
    base_health: float = 100.0
    base_energy: float = 100.0
    
    # Evolution settings
    mutation_rate: float = 0.15
    crossover_rate: float = 0.3
    elite_ratio: float = 0.2
    trait_variation: float = 0.1
    
    # Experiment settings
    num_generations: int = 50
    battles_per_generation: int = 5
    population_per_species: int = 6
    
    # Neural network settings
    hidden_size: int = 64
    learning_rate: float = 0.001
    
    # Behavior analysis
    behavior_tracking: bool = True
    save_trajectories: bool = True
    analyze_emergent_behaviors: bool = True
    
    @classmethod
    def create_fast_experiment(cls) -> 'Config':
        """Create config for fast experimentation"""
        return cls(
            max_simulation_time=120.0,
            num_generations=20,
            battles_per_generation=3,
            population_per_species=4
        )
    
    @classmethod
    def create_detailed_experiment(cls) -> 'Config':
        """Create config for detailed long-term evolution"""
        return cls(
            max_simulation_time=600.0,
            num_generations=100,
            battles_per_generation=10,
            population_per_species=8
        )

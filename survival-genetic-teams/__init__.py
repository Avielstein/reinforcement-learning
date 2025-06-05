"""
Multi-Agent Genetic Team Survival System

A sophisticated reinforcement learning environment where teams of agents 
evolve through genetic algorithms to survive in a competitive multi-agent world.
"""

__version__ = "1.0.0"
__author__ = "Reinforcement Learning Research"

# Make core components easily accessible
from .core.config import Config
from .simulation.episode_runner import EpisodeRunner
from .teams.population import Population
from .environment.survival_env import SurvivalEnvironment

__all__ = [
    'Config',
    'EpisodeRunner', 
    'Population',
    'SurvivalEnvironment'
]

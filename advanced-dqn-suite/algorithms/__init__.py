"""
Advanced DQN Suite - Algorithm Implementations

This module contains implementations of various DQN algorithms:
- BaseDQNAgent: Foundation class with vanilla DQN
- DoubleDQNAgent: Double DQN addressing overestimation bias
- More algorithms coming soon: Dueling DQN, Prioritized Replay, Rainbow DQN
"""

from .base_dqn import BaseDQNAgent, ReplayBuffer, DQNNetwork
from .double_dqn import DoubleDQNAgent, create_double_dqn_agent

__all__ = [
    'BaseDQNAgent',
    'DoubleDQNAgent', 
    'ReplayBuffer',
    'DQNNetwork',
    'create_double_dqn_agent'
]

__version__ = "0.1.0"

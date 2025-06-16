"""Double DQN agent module."""

from .double_dqn import DoubleDQN
from .networks import QNetwork
from .replay_buffer import ReplayBuffer
from .trainer import DQNTrainer

__all__ = ['DoubleDQN', 'QNetwork', 'ReplayBuffer', 'DQNTrainer']

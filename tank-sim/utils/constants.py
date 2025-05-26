"""
Constants for Fish Tank RL Environment
"""

import numpy as np

# Environment Constants
TANK_SIZE = 100.0
CENTER = np.array([TANK_SIZE/2, TANK_SIZE/2])
MAX_VELOCITY = 30.0
MAX_FORCE = 10.0
DT = 0.1
EPISODE_LEN = 200  # Shorter episodes for faster learning
VISUALIZATION_INTERVAL = 5  # More frequent updates
WALL_REPULSION = 20.0  # Reduced wall repulsion
CURRENT_COUNT = 1  # Fewer currents to start
MAX_CURRENT_STRENGTH = 5.0  # Weaker currents
CURRENT_RADIUS = 20.0

# Training parameters
GAMMA = 0.99
LAMBDA = 0.95
TRAIN_ITERS = 4
BATCH_SIZE = 512  # Smaller batches for more frequent updates
POLICY_LR = 1e-3  # Higher learning rate
VALUE_LR = 1e-3

# Random seeds for reproducibility
SEED = 42

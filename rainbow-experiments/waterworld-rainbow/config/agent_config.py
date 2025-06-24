"""RAINBOW DQN agent configuration parameters - OPTIMIZED FOR SAMPLE EFFICIENCY."""

class AgentConfig:
    """Configuration for RAINBOW DQN agent - optimized for fast learning."""
    
    # Network architecture
    HIDDEN_LAYERS = [512, 512]  # Larger networks for RAINBOW
    ACTIVATION = 'relu'
    
    # Training parameters - OPTIMIZED FOR SPEED
    LEARNING_RATE = 0.002  # Higher learning rate for faster learning on simple task
    BATCH_SIZE = 64  # Larger batch size for more stable gradients
    GAMMA = 0.99  # Discount factor
    
    # Exploration (RAINBOW uses noisy networks, so epsilon is less important)
    EPSILON_START = 1.0
    EPSILON_END = 0.01  # Lower since noisy networks handle exploration
    EPSILON_DECAY = 0.995
    
    # Experience replay - OPTIMIZED FOR FAST LEARNING
    REPLAY_BUFFER_SIZE = 20000  # Smaller buffer to start learning sooner
    MIN_REPLAY_SIZE = 500  # Start training earlier
    
    # Target network - FASTER ADAPTATION
    TARGET_UPDATE_FREQUENCY = 200  # More frequent updates for faster adaptation
    
    # Training frequency - MORE FREQUENT TRAINING
    TRAIN_FREQUENCY = 1  # Train every step for maximum sample efficiency
    
    # RAINBOW specific parameters - OPTIMIZED
    N_STEP = 5  # More aggressive multi-step learning
    V_MIN = -10.0  # Distributional RL value range
    V_MAX = 10.0
    N_ATOMS = 51  # Number of atoms for value distribution
    NOISY_STD = 0.4  # Slightly lower noise for more focused exploration
    
    # SAMPLE EFFICIENCY OPTIMIZATIONS - IMPROVED FOR RAINBOW
    EARLY_STOPPING_PATIENCE = 200  # Stop if no improvement for N episodes (increased for RAINBOW complexity)
    CONVERGENCE_THRESHOLD = 0.05  # Consider converged if improvement < this (more sensitive)
    PERFORMANCE_WINDOW = 50  # Episodes to average for performance tracking (larger window for stability)
    
    # Observation space
    # Will be set dynamically based on sensor count
    OBSERVATION_DIM = None  # Set by environment
    
    # Action space
    ACTION_DIM = 8  # 8 discrete movement directions (matches trainer mapping)
    ACTION_SCALE = 1.0  # Scale factor for actions
    USE_CONTINUOUS_ACTIONS = False  # Use discrete action space

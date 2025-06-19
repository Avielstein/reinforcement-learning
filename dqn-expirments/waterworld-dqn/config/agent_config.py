"""Double DQN agent configuration parameters."""

class AgentConfig:
    """Configuration for Double DQN agent."""
    
    # Network architecture
    HIDDEN_LAYERS = [128, 128]  # Hidden layer sizes
    ACTIVATION = 'relu'
    
    # Training parameters
    LEARNING_RATE = 0.0005  # Lower for more stable learning
    BATCH_SIZE = 32
    GAMMA = 0.99  # Discount factor
    
    # Exploration (optimized for better performance)
    EPSILON_START = 1.0
    EPSILON_END = 0.05  # Higher minimum exploration
    EPSILON_DECAY = 0.9995  # Slower decay for more exploration
    
    # Experience replay
    REPLAY_BUFFER_SIZE = 10000
    MIN_REPLAY_SIZE = 1000  # Minimum experiences before training
    
    # Target network
    TARGET_UPDATE_FREQUENCY = 100  # Steps between target network updates
    
    # Training frequency
    TRAIN_FREQUENCY = 4  # Train every N steps
    
    # Observation space
    # Will be set dynamically based on sensor count
    OBSERVATION_DIM = None  # Set by environment
    
    # Action space
    ACTION_DIM = 2  # [dx, dy] movement direction (continuous)
    ACTION_SCALE = 1.0  # Scale factor for actions
    USE_CONTINUOUS_ACTIONS = True  # Use continuous action space

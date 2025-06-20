"""WaterWorld environment configuration parameters - ACTIVE EXPLORATION VERSION."""

class EnvironmentConfig:
    """Configuration for WaterWorld environment - encourages active exploration."""
    
    # World dimensions
    WORLD_WIDTH = 500
    WORLD_HEIGHT = 500
    
    # Agent properties
    AGENT_RADIUS = 8
    AGENT_SPEED = 3
    AGENT_START_X = 250
    AGENT_START_Y = 250
    
    # Items
    GOOD_ITEM_COUNT = 8
    BAD_ITEM_COUNT = 8
    ITEM_RADIUS = 6
    GOOD_ITEM_REWARD = 5.0  # Increased reward for good items
    BAD_ITEM_REWARD = -2.0  # Reduced penalty for bad items
    
    # Sensing system
    SENSOR_COUNT = 30  # Number of raycast sensors
    SENSOR_RANGE = 120  # Maximum sensing distance
    SENSOR_ANGLE_SPAN = 2 * 3.14159  # Full circle (2π radians)
    
    # Episode settings
    MAX_EPISODE_STEPS = 1000
    TERMINATE_ON_BAD_ITEM = False  # Don't terminate - just penalty
    BAD_ITEM_PENALTY = -2.0  # Moderate penalty for hitting red items
    
    # ACTIVE EXPLORATION INCENTIVES
    MOVEMENT_REWARD = 0.01  # Small reward for moving
    IDLE_PENALTY = -0.02  # Penalty for staying in same area
    EXPLORATION_BONUS = 0.1  # Bonus for visiting new areas
    
    # Physics
    COLLISION_TOLERANCE = 1.0  # Distance for collision detection
    
    # Item respawning
    RESPAWN_DELAY = 0  # Immediate respawn
    MIN_SPAWN_DISTANCE = 30  # Minimum distance from agent when spawning

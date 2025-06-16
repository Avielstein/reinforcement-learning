"""UI configuration for research-grade interface."""

class UIConfig:
    """Configuration for user interface."""
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = 8080
    DEBUG = False
    
    # Canvas settings
    CANVAS_WIDTH = 500
    CANVAS_HEIGHT = 500
    FPS = 60
    
    # Colors (research-grade, minimal)
    BACKGROUND_COLOR = '#FFFFFF'  # White background
    AGENT_COLOR = '#000000'  # Black agent
    GOOD_ITEM_COLOR = '#00AA00'  # Green items
    BAD_ITEM_COLOR = '#AA0000'  # Red items
    SENSOR_COLOR = '#CCCCCC'  # Light gray sensors
    SENSOR_HIT_COLOR = '#666666'  # Darker gray for hits
    
    # UI Layout
    SIDEBAR_WIDTH = 300
    HEADER_HEIGHT = 60
    PANEL_MARGIN = 10
    
    # Typography (academic style)
    FONT_FAMILY = 'Arial, sans-serif'
    FONT_SIZE_NORMAL = '14px'
    FONT_SIZE_SMALL = '12px'
    FONT_SIZE_LARGE = '16px'
    
    # Chart settings
    CHART_HEIGHT = 150
    CHART_UPDATE_INTERVAL = 100  # ms
    MAX_CHART_POINTS = 200
    
    # Parameter control ranges
    PARAMETER_RANGES = {
        'learning_rate': {'min': 0.0001, 'max': 0.01, 'step': 0.0001},
        'epsilon_decay': {'min': 0.99, 'max': 0.9999, 'step': 0.0001},
        'batch_size': {'min': 16, 'max': 128, 'step': 16},
        'target_update_freq': {'min': 10, 'max': 1000, 'step': 10},
        'gamma': {'min': 0.9, 'max': 0.999, 'step': 0.001}
    }
    
    # Update frequencies
    METRICS_UPDATE_RATE = 10  # Hz
    VISUALIZATION_UPDATE_RATE = 30  # Hz

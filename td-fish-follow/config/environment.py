"""
Environment configuration for TD Fish Follow.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class EnvironmentConfig:
    """Configuration for the fish tank environment."""
    
    # Tank dimensions
    tank_width: float = 800.0
    tank_height: float = 600.0
    
    # Fish properties
    fish_size: float = 8.0
    fish_max_speed: float = 100.0
    fish_max_acceleration: float = 200.0
    fish_drag_coefficient: float = 0.95
    
    # Target properties
    target_size: float = 6.0
    target_speed: float = 50.0
    target_direction_change_prob: float = 0.02  # Probability per step
    target_smoothing: float = 0.8  # Movement smoothing factor
    
    # Physics
    dt: float = 0.016  # 60 FPS
    water_current_strength: float = 10.0
    water_current_change_rate: float = 0.001
    
    # Observation space
    observation_dim: int = 15
    normalize_observations: bool = True
    include_history: bool = True
    history_length: int = 3
    
    # Action space
    action_dim: int = 2
    action_smoothing: float = 0.7  # Temporal consistency
    max_thrust: float = 150.0
    
    # Episode settings
    max_episode_steps: int = 1000
    success_distance: float = 20.0  # Distance considered "following"
    
    # Reward structure
    distance_reward_scale: float = 1.0
    velocity_alignment_scale: float = 0.5
    smoothness_reward_scale: float = 0.1
    wall_penalty_scale: float = 0.3
    success_bonus: float = 2.0
    
    # Target movement patterns
    available_patterns: List[str] = None
    
    def __post_init__(self):
        if self.available_patterns is None:
            self.available_patterns = [
                'random_walk',
                'circular',
                'figure8',
                'zigzag',
                'spiral'
            ]
    
    def get_tank_bounds(self) -> Tuple[float, float, float, float]:
        """Get tank boundaries (min_x, max_x, min_y, max_y)."""
        return (0, self.tank_width, 0, self.tank_height)
    
    def get_center(self) -> Tuple[float, float]:
        """Get tank center coordinates."""
        return (self.tank_width / 2, self.tank_height / 2)
    
    def normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Normalize position to [-1, 1] range."""
        center_x, center_y = self.get_center()
        normalized_x = (pos[0] - center_x) / (self.tank_width / 2)
        normalized_y = (pos[1] - center_y) / (self.tank_height / 2)
        return np.array([normalized_x, normalized_y])
    
    def denormalize_position(self, norm_pos: np.ndarray) -> np.ndarray:
        """Convert normalized position back to tank coordinates."""
        center_x, center_y = self.get_center()
        x = norm_pos[0] * (self.tank_width / 2) + center_x
        y = norm_pos[1] * (self.tank_height / 2) + center_y
        return np.array([x, y])
    
    def is_valid_position(self, pos: np.ndarray, margin: float = 0) -> bool:
        """Check if position is within tank bounds."""
        min_x, max_x, min_y, max_y = self.get_tank_bounds()
        return (min_x + margin <= pos[0] <= max_x - margin and
                min_y + margin <= pos[1] <= max_y - margin)
    
    def distance_to_wall(self, pos: np.ndarray) -> float:
        """Calculate minimum distance to any wall."""
        min_x, max_x, min_y, max_y = self.get_tank_bounds()
        dist_x = min(pos[0] - min_x, max_x - pos[0])
        dist_y = min(pos[1] - min_y, max_y - pos[1])
        return min(dist_x, dist_y)

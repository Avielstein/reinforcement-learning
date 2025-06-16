"""Entity classes for WaterWorld environment."""

import math
import random
from typing import Tuple

class Agent:
    """The learning agent in WaterWorld."""
    
    def __init__(self, x: float, y: float, radius: float, speed: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.angle = 0.0  # Facing direction
        
    def move(self, dx: float, dy: float, world_width: float, world_height: float):
        """Move agent by dx, dy with boundary constraints."""
        # Update position
        self.x += dx * self.speed
        self.y += dy * self.speed
        
        # Update facing direction
        if dx != 0 or dy != 0:
            self.angle = math.atan2(dy, dx)
        
        # Keep within world boundaries
        self.x = max(self.radius, min(world_width - self.radius, self.x))
        self.y = max(self.radius, min(world_height - self.radius, self.y))
    
    def get_position(self) -> Tuple[float, float]:
        """Get agent position."""
        return self.x, self.y
    
    def distance_to(self, other_x: float, other_y: float) -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)


class Item:
    """Base class for collectible items."""
    
    def __init__(self, x: float, y: float, radius: float, reward: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.reward = reward
        self.active = True
    
    def get_position(self) -> Tuple[float, float]:
        """Get item position."""
        return self.x, self.y
    
    def collect(self) -> float:
        """Collect this item and return reward."""
        if self.active:
            self.active = False
            return self.reward
        return 0.0
    
    def respawn(self, x: float, y: float):
        """Respawn item at new position."""
        self.x = x
        self.y = y
        self.active = True


class GoodItem(Item):
    """Green item that gives positive reward."""
    
    def __init__(self, x: float, y: float, radius: float, reward: float = 1.0):
        super().__init__(x, y, radius, reward)


class BadItem(Item):
    """Red item that gives negative reward."""
    
    def __init__(self, x: float, y: float, radius: float, reward: float = -1.0):
        super().__init__(x, y, radius, reward)


def spawn_random_position(world_width: float, world_height: float, 
                         avoid_x: float, avoid_y: float, min_distance: float,
                         item_radius: float) -> Tuple[float, float]:
    """Generate random spawn position avoiding agent."""
    max_attempts = 100
    
    for _ in range(max_attempts):
        x = random.uniform(item_radius, world_width - item_radius)
        y = random.uniform(item_radius, world_height - item_radius)
        
        # Check distance from agent
        distance = math.sqrt((x - avoid_x)**2 + (y - avoid_y)**2)
        if distance >= min_distance:
            return x, y
    
    # Fallback: place at corner
    return item_radius, item_radius

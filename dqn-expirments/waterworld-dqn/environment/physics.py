"""Physics engine for WaterWorld collision detection."""

import math
from typing import List, Tuple
from .entities import Agent, Item

class PhysicsEngine:
    """Handles collision detection and physics interactions."""
    
    def __init__(self, collision_tolerance: float = 1.0):
        self.collision_tolerance = collision_tolerance
    
    def check_collisions(self, agent: Agent, items: List[Item]) -> List[Tuple[Item, float]]:
        """Check collisions between agent and items. Returns list of (item, reward) pairs."""
        collisions = []
        
        for item in items:
            if item.active and self._circles_collide(agent, item):
                reward = item.collect()
                if reward != 0:  # Only add if item was actually collected
                    collisions.append((item, reward))
        
        return collisions
    
    def _circles_collide(self, agent: Agent, item: Item) -> bool:
        """Check if two circular objects collide."""
        distance = agent.distance_to(item.x, item.y)
        collision_distance = agent.radius + item.radius + self.collision_tolerance
        return distance <= collision_distance
    
    def normalize_action(self, action: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize action vector to unit length."""
        dx, dy = action
        magnitude = math.sqrt(dx*dx + dy*dy)
        
        if magnitude == 0:
            return 0.0, 0.0
        
        return dx / magnitude, dy / magnitude
    
    def apply_boundaries(self, agent: Agent, world_width: float, world_height: float):
        """Ensure agent stays within world boundaries."""
        agent.x = max(agent.radius, min(world_width - agent.radius, agent.x))
        agent.y = max(agent.radius, min(world_height - agent.radius, agent.y))
    
    def get_boundary_distance(self, agent: Agent, world_width: float, world_height: float) -> dict:
        """Get distances to each boundary."""
        return {
            'left': agent.x - agent.radius,
            'right': world_width - agent.x - agent.radius,
            'top': agent.y - agent.radius,
            'bottom': world_height - agent.y - agent.radius
        }

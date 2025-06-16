"""Main WaterWorld environment implementation."""

import random
import numpy as np
from typing import List, Tuple, Dict, Any

from config import EnvironmentConfig
from .entities import Agent, GoodItem, BadItem, spawn_random_position
from .sensors import SensorSystem, SensorReading
from .physics import PhysicsEngine

class WaterWorld:
    """WaterWorld environment for reinforcement learning."""
    
    def __init__(self, config: EnvironmentConfig = None):
        self.config = config or EnvironmentConfig()
        
        # Initialize components
        self.agent = Agent(
            self.config.AGENT_START_X,
            self.config.AGENT_START_Y,
            self.config.AGENT_RADIUS,
            self.config.AGENT_SPEED
        )
        
        self.sensor_system = SensorSystem(
            self.config.SENSOR_COUNT,
            self.config.SENSOR_RANGE,
            self.config.SENSOR_ANGLE_SPAN
        )
        
        self.physics = PhysicsEngine(self.config.COLLISION_TOLERANCE)
        
        # Initialize items
        self.good_items: List[GoodItem] = []
        self.bad_items: List[BadItem] = []
        self._spawn_items()
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.total_episodes = 0
        
        # Metrics
        self.items_collected = 0
        self.items_avoided = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Reset agent position
        self.agent.x = self.config.AGENT_START_X
        self.agent.y = self.config.AGENT_START_Y
        self.agent.angle = 0.0
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.total_episodes += 1
        
        # Reset metrics
        self.items_collected = 0
        self.items_avoided = 0
        
        # Respawn all items
        self._spawn_items()
        
        return self._get_observation()
    
    def step(self, action: Tuple[float, float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Normalize and apply action
        dx, dy = self.physics.normalize_action(action)
        self.agent.move(dx, dy, self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT)
        
        # Check collisions and calculate reward
        reward = 0.0
        
        # Check good items
        good_collisions = self.physics.check_collisions(self.agent, self.good_items)
        for item, item_reward in good_collisions:
            reward += item_reward
            self.items_collected += 1
            self._respawn_item(item)
        
        # Check bad items
        bad_collisions = self.physics.check_collisions(self.agent, self.bad_items)
        for item, item_reward in bad_collisions:
            reward += item_reward
            self.items_avoided += 1
            self._respawn_item(item)
        
        # Update episode tracking
        self.episode_step += 1
        self.episode_reward += reward
        
        # Check if episode is done
        done = self.episode_step >= self.config.MAX_EPISODE_STEPS
        
        # Get new observation
        observation = self._get_observation()
        
        # Create info dict
        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'items_collected': self.items_collected,
            'items_avoided': self.items_avoided,
            'agent_position': (self.agent.x, self.agent.y),
            'agent_angle': self.agent.angle
        }
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        # Get sensor readings
        sensor_readings = self.sensor_system.sense(
            self.agent, self.good_items, self.bad_items,
            self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT
        )
        
        # Convert to observation vector
        # For each sensor: [distance, is_good, is_bad, is_wall]
        observation = []
        
        for reading in sensor_readings:
            observation.append(reading.distance)
            observation.append(1.0 if reading.item_type == 'good' else 0.0)
            observation.append(1.0 if reading.item_type == 'bad' else 0.0)
            observation.append(1.0 if reading.item_type == 'wall' else 0.0)
        
        return np.array(observation, dtype=np.float32)
    
    def get_observation_dim(self) -> int:
        """Get observation space dimension."""
        return self.config.SENSOR_COUNT * 4  # distance + 3 type flags per sensor
    
    def get_sensor_readings(self) -> List[SensorReading]:
        """Get current sensor readings for visualization."""
        return self.sensor_system.sense(
            self.agent, self.good_items, self.bad_items,
            self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT
        )
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete environment state for visualization."""
        return {
            'agent': {
                'x': self.agent.x,
                'y': self.agent.y,
                'angle': self.agent.angle,
                'radius': self.agent.radius
            },
            'good_items': [
                {'x': item.x, 'y': item.y, 'radius': item.radius, 'active': item.active}
                for item in self.good_items
            ],
            'bad_items': [
                {'x': item.x, 'y': item.y, 'radius': item.radius, 'active': item.active}
                for item in self.bad_items
            ],
            'sensor_readings': [
                {
                    'distance': reading.distance,
                    'type': reading.item_type,
                    'hit_position': reading.hit_position
                }
                for reading in self.get_sensor_readings()
            ],
            'world_size': (self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT)
        }
    
    def _spawn_items(self):
        """Spawn all items in random positions."""
        self.good_items.clear()
        self.bad_items.clear()
        
        # Spawn good items
        for _ in range(self.config.GOOD_ITEM_COUNT):
            x, y = spawn_random_position(
                self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT,
                self.agent.x, self.agent.y, self.config.MIN_SPAWN_DISTANCE,
                self.config.ITEM_RADIUS
            )
            item = GoodItem(x, y, self.config.ITEM_RADIUS, self.config.GOOD_ITEM_REWARD)
            self.good_items.append(item)
        
        # Spawn bad items
        for _ in range(self.config.BAD_ITEM_COUNT):
            x, y = spawn_random_position(
                self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT,
                self.agent.x, self.agent.y, self.config.MIN_SPAWN_DISTANCE,
                self.config.ITEM_RADIUS
            )
            item = BadItem(x, y, self.config.ITEM_RADIUS, self.config.BAD_ITEM_REWARD)
            self.bad_items.append(item)
    
    def _respawn_item(self, item):
        """Respawn a collected item at new random position."""
        x, y = spawn_random_position(
            self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT,
            self.agent.x, self.agent.y, self.config.MIN_SPAWN_DISTANCE,
            self.config.ITEM_RADIUS
        )
        item.respawn(x, y)

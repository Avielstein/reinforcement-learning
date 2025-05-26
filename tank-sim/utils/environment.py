"""
Fish Tank Environment and Water Current Classes
"""

import numpy as np
import math
from .constants import *


class WaterCurrent:
    """Represents a water current in the environment"""
    
    def __init__(self):
        self.position = np.random.uniform(0, TANK_SIZE, 2)
        self.direction = self._random_direction()
        self.strength = np.random.uniform(3.0, MAX_CURRENT_STRENGTH)
        self.radius = CURRENT_RADIUS
        
    def _random_direction(self):
        """Generate a random unit vector for direction"""
        vec = np.random.normal(size=2)
        return vec / np.linalg.norm(vec)
    
    def vector(self, pos):
        """Calculate force vector at given position"""
        d = self.position - pos
        dist = np.linalg.norm(d)
        if dist > self.radius:
            return np.zeros(2)
        return self.direction * self.strength * (1 - dist/self.radius)
    
    def step(self):
        """Update current position and direction"""
        # Slowly drift the current
        self.position = (self.position + np.random.normal(0, 0.3, 2)) % TANK_SIZE
        # Slowly change direction
        self.direction += 0.05 * self._random_direction()
        self.direction /= np.linalg.norm(self.direction)


class FishTankEnv:
    """Fish environment with water currents"""
    
    obs_dim = 4  # position (relative to center) + velocity
    act_dim = 2  # thrust in x, y directions
    
    def __init__(self):
        self.currents = [WaterCurrent() for _ in range(CURRENT_COUNT)]
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Start fish at random position
        self.position = np.random.uniform(0, TANK_SIZE, 2)
        self.velocity = np.zeros(2)
        self.step_count = 0
        return self._obs()
    
    def step(self, action):
        """Take a step in the environment"""
        # Clip and scale action
        action = np.clip(action, -1, 1) * MAX_FORCE
        
        # Calculate total force
        force = action.copy()
        
        # Add current forces
        for cur in self.currents:
            force += cur.vector(self.position)
        
        # Add wall repulsion
        force += self._wall_force()
        
        # Update velocity
        self.velocity += force * DT
        
        # Limit velocity
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_VELOCITY:
            self.velocity *= MAX_VELOCITY / speed
        
        # Update position (with proper boundary handling)
        self.position = self.position + self.velocity * DT
        
        # Keep fish within bounds (no wrapping)
        self.position = np.clip(self.position, 0, TANK_SIZE)
        
        # Update currents occasionally
        if self.step_count % 5 == 0:
            for c in self.currents:
                c.step()
        
        self.step_count += 1
        done = self.step_count >= EPISODE_LEN
        
        return self._obs(), self._reward(), done, {}
    
    def _obs(self):
        """Get observation vector"""
        # Position relative to center, normalized
        rel_pos = (self.position - CENTER) / (TANK_SIZE/2)
        # Velocity normalized
        norm_vel = self.velocity / MAX_VELOCITY
        return np.concatenate([rel_pos, norm_vel]).astype(np.float32)
    
    def _wall_force(self):
        """Calculate wall repulsion force"""
        w = np.zeros(2)
        
        # X direction
        if self.position[0] < 1:
            w[0] = WALL_REPULSION * (1 - self.position[0])
        elif self.position[0] > TANK_SIZE - 1:
            w[0] = WALL_REPULSION * (TANK_SIZE - 1 - self.position[0])
        
        # Y direction
        if self.position[1] < 1:
            w[1] = WALL_REPULSION * (1 - self.position[1])
        elif self.position[1] > TANK_SIZE - 1:
            w[1] = WALL_REPULSION * (TANK_SIZE - 1 - self.position[1])
        
        return w
    
    def _reward(self):
        """Calculate reward"""
        # Distance to center
        d = np.linalg.norm(self.position - CENTER)
        max_dist = TANK_SIZE / 2 * math.sqrt(2)  # Maximum possible distance
        
        # Linear reward for being close to center (easier to learn)
        position_reward = 1.0 - (d / max_dist)
        
        # Bonus for being very close to center
        if d < 5.0:
            position_reward += 2.0
        elif d < 10.0:
            position_reward += 1.0
        
        # Small penalty for high velocity (encourage stability)
        velocity_penalty = -0.01 * np.linalg.norm(self.velocity) / MAX_VELOCITY
        
        # Penalty for being near walls
        wall_penalty = 0.0
        margin = 5.0
        if (self.position[0] < margin or self.position[0] > TANK_SIZE - margin or
            self.position[1] < margin or self.position[1] > TANK_SIZE - margin):
            wall_penalty = -0.5
        
        return position_reward + velocity_penalty + wall_penalty

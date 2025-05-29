"""
Dot Follow Environment - Fish learns to follow a moving target
"""

import numpy as np
import math
import sys
sys.path.append('../')
from utils.constants import *


class MovingTarget:
    """Represents a moving target that the fish must follow"""
    
    def __init__(self, movement_pattern='circular'):
        self.position = np.array([TANK_SIZE/2, TANK_SIZE/2], dtype=np.float32)
        self.movement_pattern = movement_pattern
        self.time = 0.0
        self.speed = 8.0  # Slower target movement speed for easier learning
        self.radius = 20.0  # Smaller radius for easier tracking
        self.direction = np.random.uniform(0, 2*np.pi)  # Random initial direction
        self.direction_change_prob = 0.01  # Less frequent direction changes
        
    def step(self):
        """Update target position based on movement pattern"""
        self.time += DT
        
        if self.movement_pattern == 'circular':
            # Move in a circle around the center
            center_x, center_y = TANK_SIZE/2, TANK_SIZE/2
            angle = self.time * self.speed / self.radius
            self.position[0] = center_x + self.radius * np.cos(angle)
            self.position[1] = center_y + self.radius * np.sin(angle)
            
        elif self.movement_pattern == 'figure8':
            # Move in a figure-8 pattern
            center_x, center_y = TANK_SIZE/2, TANK_SIZE/2
            t = self.time * self.speed / 20.0
            self.position[0] = center_x + self.radius * np.sin(t)
            self.position[1] = center_y + self.radius * np.sin(2*t) / 2
            
        elif self.movement_pattern == 'random_walk':
            # Random walk with smooth direction changes
            if np.random.random() < self.direction_change_prob:
                self.direction += np.random.normal(0, 0.3)
            
            # Move in current direction
            dx = self.speed * np.cos(self.direction) * DT
            dy = self.speed * np.sin(self.direction) * DT
            
            new_pos = self.position + np.array([dx, dy])
            
            # Bounce off walls
            if new_pos[0] < 10 or new_pos[0] > TANK_SIZE - 10:
                self.direction = np.pi - self.direction
            if new_pos[1] < 10 or new_pos[1] > TANK_SIZE - 10:
                self.direction = -self.direction
                
            # Keep within bounds
            self.position = np.clip(new_pos, 10, TANK_SIZE - 10)
            
        elif self.movement_pattern == 'zigzag':
            # Zigzag pattern across the tank
            t = self.time * self.speed / 10.0
            self.position[0] = TANK_SIZE/2 + self.radius * np.sin(t)
            self.position[1] = 20 + (TANK_SIZE - 40) * (0.5 + 0.5 * np.sin(t/3))
            
        elif self.movement_pattern == 'spiral':
            # Spiral pattern
            t = self.time * self.speed / 15.0
            r = 10 + (self.radius * t / (2*np.pi))
            self.position[0] = TANK_SIZE/2 + r * np.cos(t)
            self.position[1] = TANK_SIZE/2 + r * np.sin(t)
            
            # Reset spiral when it gets too big
            if r > TANK_SIZE/2 - 10:
                self.time = 0.0
        
        # Ensure target stays within bounds
        self.position = np.clip(self.position, 5, TANK_SIZE - 5)


class WaterCurrent:
    """Water current class - simplified for dot follow scenario"""
    
    def __init__(self):
        self.position = np.random.uniform(0, TANK_SIZE, 2)
        self.direction = self._random_direction()
        self.strength = np.random.uniform(1.0, 3.0)  # Weaker currents
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
        self.position = (self.position + np.random.normal(0, 0.2, 2)) % TANK_SIZE
        # Slowly change direction
        self.direction += 0.03 * self._random_direction()
        self.direction /= np.linalg.norm(self.direction)


class DotFollowEnv:
    """Fish environment where the goal is to follow a moving target"""
    
    obs_dim = 6  # fish position, fish velocity, target position (all relative to center)
    act_dim = 2  # thrust in x, y directions
    
    def __init__(self, movement_pattern='circular'):
        self.target = MovingTarget(movement_pattern)
        self.currents = [WaterCurrent() for _ in range(max(1, CURRENT_COUNT//2))]  # Fewer currents
        self.movement_pattern = movement_pattern
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Start fish at random position
        self.position = np.random.uniform(20, TANK_SIZE-20, 2)
        self.velocity = np.zeros(2)
        self.step_count = 0
        
        # Reset target
        self.target = MovingTarget(self.movement_pattern)
        
        return self._obs()
    
    def step(self, action):
        """Take a step in the environment"""
        # Clip and scale action
        action = np.clip(action, -1, 1) * MAX_FORCE
        
        # Calculate total force
        force = action.copy()
        
        # Add current forces (reduced influence)
        for cur in self.currents:
            force += cur.vector(self.position) * 0.5  # Reduced current strength
        
        # Add wall repulsion
        force += self._wall_force()
        
        # Update velocity
        self.velocity += force * DT
        
        # Limit velocity
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_VELOCITY:
            self.velocity *= MAX_VELOCITY / speed
        
        # Update position
        self.position = self.position + self.velocity * DT
        
        # Keep fish within bounds
        self.position = np.clip(self.position, 0, TANK_SIZE)
        
        # Update target position
        self.target.step()
        
        # Update currents occasionally
        if self.step_count % 10 == 0:
            for c in self.currents:
                c.step()
        
        self.step_count += 1
        done = self.step_count >= EPISODE_LEN
        
        return self._obs(), self._reward(), done, {'target_pos': self.target.position.copy()}
    
    def _obs(self):
        """Get observation vector"""
        # Fish position relative to center, normalized
        fish_rel_pos = (self.position - CENTER) / (TANK_SIZE/2)
        
        # Fish velocity normalized
        norm_vel = self.velocity / MAX_VELOCITY
        
        # Target position relative to center, normalized
        target_rel_pos = (self.target.position - CENTER) / (TANK_SIZE/2)
        
        return np.concatenate([fish_rel_pos, norm_vel, target_rel_pos]).astype(np.float32)
    
    def _wall_force(self):
        """Calculate wall repulsion force"""
        w = np.zeros(2)
        margin = 5.0
        
        # X direction
        if self.position[0] < margin:
            w[0] = WALL_REPULSION * (margin - self.position[0]) / margin
        elif self.position[0] > TANK_SIZE - margin:
            w[0] = WALL_REPULSION * (TANK_SIZE - margin - self.position[0]) / margin
        
        # Y direction
        if self.position[1] < margin:
            w[1] = WALL_REPULSION * (margin - self.position[1]) / margin
        elif self.position[1] > TANK_SIZE - margin:
            w[1] = WALL_REPULSION * (TANK_SIZE - margin - self.position[1]) / margin
        
        return w
    
    def _reward(self):
        """Calculate reward based on proximity to moving target"""
        # Distance to target
        target_dist = np.linalg.norm(self.position - self.target.position)
        
        # Primary reward: exponential reward for being close to target
        max_dist = TANK_SIZE * math.sqrt(2)  # Maximum possible distance
        normalized_dist = target_dist / max_dist
        
        # Exponential reward - much higher rewards for being close
        proximity_reward = 10.0 * np.exp(-5.0 * normalized_dist)
        
        # Additional bonuses for being very close to target
        if target_dist < 3.0:
            proximity_reward += 5.0
        elif target_dist < 8.0:
            proximity_reward += 2.0
        elif target_dist < 15.0:
            proximity_reward += 1.0
        
        # Small penalty for high velocity (encourage smooth following)
        velocity_penalty = -0.1 * np.linalg.norm(self.velocity) / MAX_VELOCITY
        
        # Penalty for being near walls
        wall_penalty = 0.0
        margin = 8.0
        if (self.position[0] < margin or self.position[0] > TANK_SIZE - margin or
            self.position[1] < margin or self.position[1] > TANK_SIZE - margin):
            wall_penalty = -0.3
        
        # Bonus for following target movement (reward for being in the right direction)
        if hasattr(self, '_prev_target_dist'):
            if target_dist < self._prev_target_dist:
                proximity_reward += 0.2  # Bonus for getting closer
        self._prev_target_dist = target_dist
        
        return proximity_reward + velocity_penalty + wall_penalty
    
    def get_target_position(self):
        """Get current target position"""
        return self.target.position.copy()
    
    def set_movement_pattern(self, pattern):
        """Change the target movement pattern"""
        self.movement_pattern = pattern
        self.target.movement_pattern = pattern

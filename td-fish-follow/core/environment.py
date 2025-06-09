"""
Enhanced fish tank environment with rich observations for TD learning.
"""

import numpy as np
import math
from typing import Tuple, Dict, Any, Optional, List
from collections import deque

from ..config.environment import EnvironmentConfig


class MovingTarget:
    """Moving target with various movement patterns."""
    
    def __init__(self, config: EnvironmentConfig, pattern: str = 'random_walk'):
        self.config = config
        self.pattern = pattern
        self.position = np.array([config.tank_width / 2, config.tank_height / 2])
        self.velocity = np.zeros(2)
        self.direction = 0.0
        self.time = 0.0
        self.pattern_params = {}
        
        # Initialize pattern-specific parameters
        self._init_pattern()
    
    def _init_pattern(self):
        """Initialize pattern-specific parameters."""
        center_x, center_y = self.config.get_center()
        
        if self.pattern == 'circular':
            self.pattern_params = {
                'center': np.array([center_x, center_y]),
                'radius': min(self.config.tank_width, self.config.tank_height) * 0.3,
                'angular_speed': 0.02
            }
        elif self.pattern == 'figure8':
            self.pattern_params = {
                'center': np.array([center_x, center_y]),
                'scale': min(self.config.tank_width, self.config.tank_height) * 0.25,
                'speed': 0.03
            }
        elif self.pattern == 'zigzag':
            self.pattern_params = {
                'amplitude': self.config.tank_height * 0.3,
                'frequency': 0.01,
                'speed': self.config.target_speed
            }
        elif self.pattern == 'spiral':
            self.pattern_params = {
                'center': np.array([center_x, center_y]),
                'max_radius': min(self.config.tank_width, self.config.tank_height) * 0.4,
                'angular_speed': 0.03,
                'radial_speed': 0.5
            }
        elif self.pattern == 'random_walk':
            self.pattern_params = {
                'direction_change_prob': self.config.target_direction_change_prob,
                'smoothing': self.config.target_smoothing
            }
    
    def step(self) -> np.ndarray:
        """Update target position based on movement pattern."""
        self.time += self.config.dt
        
        if self.pattern == 'circular':
            self._update_circular()
        elif self.pattern == 'figure8':
            self._update_figure8()
        elif self.pattern == 'zigzag':
            self._update_zigzag()
        elif self.pattern == 'spiral':
            self._update_spiral()
        elif self.pattern == 'random_walk':
            self._update_random_walk()
        
        # Ensure target stays within bounds
        self._clamp_to_bounds()
        
        return self.position.copy()
    
    def _update_circular(self):
        """Update position for circular movement."""
        params = self.pattern_params
        angle = self.time * params['angular_speed']
        self.position = params['center'] + params['radius'] * np.array([
            np.cos(angle), np.sin(angle)
        ])
        self.velocity = params['radius'] * params['angular_speed'] * np.array([
            -np.sin(angle), np.cos(angle)
        ])
    
    def _update_figure8(self):
        """Update position for figure-8 movement."""
        params = self.pattern_params
        t = self.time * params['speed']
        scale = params['scale']
        
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)
        
        self.position = params['center'] + np.array([x, y])
        
        # Calculate velocity
        dx_dt = scale * np.cos(t) * params['speed']
        dy_dt = scale * (np.cos(t)**2 - np.sin(t)**2) * params['speed']
        self.velocity = np.array([dx_dt, dy_dt])
    
    def _update_zigzag(self):
        """Update position for zigzag movement."""
        params = self.pattern_params
        
        # Move horizontally with vertical oscillation
        self.position[0] += params['speed'] * self.config.dt
        self.position[1] = (self.config.tank_height / 2 + 
                           params['amplitude'] * np.sin(self.time * params['frequency']))
        
        self.velocity = np.array([
            params['speed'],
            params['amplitude'] * params['frequency'] * np.cos(self.time * params['frequency'])
        ])
        
        # Reset when reaching edge
        if self.position[0] > self.config.tank_width:
            self.position[0] = 0
    
    def _update_spiral(self):
        """Update position for spiral movement."""
        params = self.pattern_params
        angle = self.time * params['angular_speed']
        radius = min(params['radial_speed'] * self.time, params['max_radius'])
        
        if radius >= params['max_radius']:
            # Reset spiral
            self.time = 0
            radius = 0
        
        self.position = params['center'] + radius * np.array([
            np.cos(angle), np.sin(angle)
        ])
        
        # Calculate velocity
        dr_dt = params['radial_speed'] if radius < params['max_radius'] else 0
        self.velocity = np.array([
            dr_dt * np.cos(angle) - radius * params['angular_speed'] * np.sin(angle),
            dr_dt * np.sin(angle) + radius * params['angular_speed'] * np.cos(angle)
        ])
    
    def _update_random_walk(self):
        """Update position for random walk movement."""
        params = self.pattern_params
        
        # Randomly change direction
        if np.random.random() < params['direction_change_prob']:
            self.direction = np.random.uniform(0, 2 * np.pi)
        
        # Smooth direction changes
        target_velocity = self.config.target_speed * np.array([
            np.cos(self.direction), np.sin(self.direction)
        ])
        
        self.velocity = (params['smoothing'] * self.velocity + 
                        (1 - params['smoothing']) * target_velocity)
        
        # Update position
        self.position += self.velocity * self.config.dt
    
    def _clamp_to_bounds(self):
        """Keep target within tank bounds."""
        margin = self.config.target_size
        self.position[0] = np.clip(self.position[0], margin, 
                                  self.config.tank_width - margin)
        self.position[1] = np.clip(self.position[1], margin, 
                                  self.config.tank_height - margin)
    
    def get_predicted_position(self, steps_ahead: int = 3) -> np.ndarray:
        """Predict target position several steps ahead."""
        if self.pattern == 'random_walk':
            # For random walk, just extrapolate current velocity
            return self.position + self.velocity * self.config.dt * steps_ahead
        else:
            # For deterministic patterns, we can predict accurately
            future_time = self.time + self.config.dt * steps_ahead
            saved_time = self.time
            self.time = future_time
            
            if self.pattern == 'circular':
                params = self.pattern_params
                angle = self.time * params['angular_speed']
                predicted_pos = params['center'] + params['radius'] * np.array([
                    np.cos(angle), np.sin(angle)
                ])
            elif self.pattern == 'figure8':
                params = self.pattern_params
                t = self.time * params['speed']
                scale = params['scale']
                x = scale * np.sin(t)
                y = scale * np.sin(t) * np.cos(t)
                predicted_pos = params['center'] + np.array([x, y])
            else:
                # Fallback to velocity extrapolation
                predicted_pos = self.position + self.velocity * self.config.dt * steps_ahead
            
            self.time = saved_time
            return predicted_pos


class TDFishEnvironment:
    """Enhanced fish tank environment for TD learning."""
    
    def __init__(self, config: EnvironmentConfig, pattern: str = 'random_walk'):
        self.config = config
        self.pattern = pattern
        
        # Fish state
        self.fish_position = np.zeros(2)
        self.fish_velocity = np.zeros(2)
        self.fish_acceleration = np.zeros(2)
        self.previous_action = np.zeros(2)
        
        # Target
        self.target = MovingTarget(config, pattern)
        
        # Environment state
        self.water_current = np.zeros(2)
        self.step_count = 0
        self.episode_reward = 0.0
        
        # History for temporal features
        self.position_history = deque(maxlen=config.history_length)
        self.target_history = deque(maxlen=config.history_length)
        self.distance_history = deque(maxlen=config.history_length)
        
        # Performance tracking
        self.distances_to_target = []
        self.success_count = 0
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Reset fish to random position
        margin = self.config.fish_size * 2
        self.fish_position = np.array([
            np.random.uniform(margin, self.config.tank_width - margin),
            np.random.uniform(margin, self.config.tank_height - margin)
        ])
        
        self.fish_velocity = np.zeros(2)
        self.fish_acceleration = np.zeros(2)
        self.previous_action = np.zeros(2)
        
        # Reset target
        self.target = MovingTarget(self.config, self.pattern)
        
        # Reset environment
        self.water_current = np.random.uniform(-1, 1, 2) * self.config.water_current_strength
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Clear history
        self.position_history.clear()
        self.target_history.clear()
        self.distance_history.clear()
        
        # Initialize history with current state
        for _ in range(self.config.history_length):
            self.position_history.append(self.fish_position.copy())
            self.target_history.append(self.target.position.copy())
            self.distance_history.append(self._get_distance_to_target())
        
        self.distances_to_target = []
        self.success_count = 0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Clip and smooth actions
        action = np.clip(action, -1, 1)
        smoothed_action = (self.config.action_smoothing * self.previous_action + 
                          (1 - self.config.action_smoothing) * action)
        self.previous_action = smoothed_action.copy()
        
        # Apply action to fish
        thrust = smoothed_action * self.config.max_thrust
        self._update_fish_physics(thrust)
        
        # Update target
        self.target.step()
        
        # Update water current
        self._update_water_current()
        
        # Update history
        self.position_history.append(self.fish_position.copy())
        self.target_history.append(self.target.position.copy())
        distance = self._get_distance_to_target()
        self.distance_history.append(distance)
        self.distances_to_target.append(distance)
        
        # Calculate reward
        reward = self._calculate_reward(smoothed_action)
        self.episode_reward += reward
        
        # Check if episode is done
        done = (self.step_count >= self.config.max_episode_steps or
                not self.config.is_valid_position(self.fish_position, self.config.fish_size))
        
        # Update step count
        self.step_count += 1
        
        # Create info dict
        info = {
            'distance_to_target': distance,
            'episode_reward': self.episode_reward,
            'success_rate': self.success_count / max(1, len(self.distances_to_target)),
            'fish_position': self.fish_position.copy(),
            'target_position': self.target.position.copy(),
            'fish_velocity': self.fish_velocity.copy(),
            'target_velocity': self.target.velocity.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _update_fish_physics(self, thrust: np.ndarray):
        """Update fish position and velocity based on physics."""
        # Apply thrust and water current
        total_force = thrust + self.water_current
        
        # Update acceleration (F = ma, assume mass = 1)
        self.fish_acceleration = total_force
        
        # Update velocity with drag
        self.fish_velocity += self.fish_acceleration * self.config.dt
        self.fish_velocity *= self.config.fish_drag_coefficient
        
        # Limit maximum speed
        speed = np.linalg.norm(self.fish_velocity)
        if speed > self.config.fish_max_speed:
            self.fish_velocity = (self.fish_velocity / speed) * self.config.fish_max_speed
        
        # Update position
        self.fish_position += self.fish_velocity * self.config.dt
        
        # Handle wall collisions
        self._handle_wall_collisions()
    
    def _handle_wall_collisions(self):
        """Handle fish collisions with tank walls."""
        margin = self.config.fish_size
        
        # X boundaries
        if self.fish_position[0] < margin:
            self.fish_position[0] = margin
            self.fish_velocity[0] = abs(self.fish_velocity[0]) * 0.5  # Bounce with energy loss
        elif self.fish_position[0] > self.config.tank_width - margin:
            self.fish_position[0] = self.config.tank_width - margin
            self.fish_velocity[0] = -abs(self.fish_velocity[0]) * 0.5
        
        # Y boundaries
        if self.fish_position[1] < margin:
            self.fish_position[1] = margin
            self.fish_velocity[1] = abs(self.fish_velocity[1]) * 0.5
        elif self.fish_position[1] > self.config.tank_height - margin:
            self.fish_position[1] = self.config.tank_height - margin
            self.fish_velocity[1] = -abs(self.fish_velocity[1]) * 0.5
    
    def _update_water_current(self):
        """Update water current with slow random changes."""
        change = np.random.normal(0, self.config.water_current_change_rate, 2)
        self.water_current += change
        self.water_current = np.clip(self.water_current, 
                                   -self.config.water_current_strength,
                                   self.config.water_current_strength)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = []
        
        # Fish state (6D): position, velocity, acceleration
        if self.config.normalize_observations:
            fish_pos_norm = self.config.normalize_position(self.fish_position)
            fish_vel_norm = self.fish_velocity / self.config.fish_max_speed
            fish_acc_norm = self.fish_acceleration / self.config.fish_max_acceleration
        else:
            fish_pos_norm = self.fish_position
            fish_vel_norm = self.fish_velocity
            fish_acc_norm = self.fish_acceleration
        
        obs.extend(fish_pos_norm)
        obs.extend(fish_vel_norm)
        obs.extend(fish_acc_norm)
        
        # Target state (4D): position, velocity
        if self.config.normalize_observations:
            target_pos_norm = self.config.normalize_position(self.target.position)
            target_vel_norm = self.target.velocity / self.config.target_speed
        else:
            target_pos_norm = self.target.position
            target_vel_norm = self.target.velocity
        
        obs.extend(target_pos_norm)
        obs.extend(target_vel_norm)
        
        # Relative features (3D): distance, angle, relative speed
        distance = self._get_distance_to_target()
        angle = self._get_angle_to_target()
        relative_speed = self._get_relative_speed()
        
        if self.config.normalize_observations:
            max_distance = np.sqrt(self.config.tank_width**2 + self.config.tank_height**2)
            distance_norm = distance / max_distance
            angle_norm = angle / np.pi  # Normalize to [-1, 1]
            relative_speed_norm = relative_speed / (self.config.fish_max_speed + self.config.target_speed)
        else:
            distance_norm = distance
            angle_norm = angle
            relative_speed_norm = relative_speed
        
        obs.extend([distance_norm, angle_norm, relative_speed_norm])
        
        # Environmental features (2D): wall distance, water current strength
        wall_distance = self.config.distance_to_wall(self.fish_position)
        current_strength = np.linalg.norm(self.water_current)
        
        if self.config.normalize_observations:
            max_wall_distance = min(self.config.tank_width, self.config.tank_height) / 2
            wall_distance_norm = wall_distance / max_wall_distance
            current_strength_norm = current_strength / self.config.water_current_strength
        else:
            wall_distance_norm = wall_distance
            current_strength_norm = current_strength
        
        obs.extend([wall_distance_norm, current_strength_norm])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_distance_to_target(self) -> float:
        """Calculate distance from fish to target."""
        return np.linalg.norm(self.fish_position - self.target.position)
    
    def _get_angle_to_target(self) -> float:
        """Calculate angle from fish to target."""
        diff = self.target.position - self.fish_position
        return np.arctan2(diff[1], diff[0])
    
    def _get_relative_speed(self) -> float:
        """Calculate relative speed between fish and target."""
        relative_velocity = self.fish_velocity - self.target.velocity
        return np.linalg.norm(relative_velocity)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current state and action."""
        distance = self._get_distance_to_target()
        
        # Distance-based reward (closer is better)
        max_distance = np.sqrt(self.config.tank_width**2 + self.config.tank_height**2)
        distance_reward = (1.0 - distance / max_distance) * self.config.distance_reward_scale
        
        # Velocity alignment reward (moving towards target)
        if len(self.distance_history) >= 2:
            prev_distance = self.distance_history[-2]
            if distance < prev_distance:
                distance_reward += 0.2  # Bonus for getting closer
        
        # Success bonus
        success_bonus = 0.0
        if distance < self.config.success_distance:
            success_bonus = self.config.success_bonus
            self.success_count += 1
        
        # Smoothness reward (penalize erratic movements)
        action_change = np.linalg.norm(action - self.previous_action)
        smoothness_reward = -action_change * self.config.smoothness_reward_scale
        
        # Wall penalty
        wall_distance = self.config.distance_to_wall(self.fish_position)
        wall_penalty = 0.0
        if wall_distance < self.config.fish_size * 2:
            wall_penalty = -self.config.wall_penalty_scale * (1.0 - wall_distance / (self.config.fish_size * 2))
        
        total_reward = distance_reward + success_bonus + smoothness_reward + wall_penalty
        
        return total_reward
    
    def get_state_for_td(self) -> Dict[str, Any]:
        """Get state information specifically for TD learning."""
        return {
            'observation': self._get_observation(),
            'distance_to_target': self._get_distance_to_target(),
            'fish_position': self.fish_position.copy(),
            'target_position': self.target.position.copy(),
            'fish_velocity': self.fish_velocity.copy(),
            'target_velocity': self.target.velocity.copy(),
            'step_count': self.step_count,
            'predicted_target_position': self.target.get_predicted_position()
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render environment (placeholder for visualization)."""
        # This will be implemented in the visualization module
        pass

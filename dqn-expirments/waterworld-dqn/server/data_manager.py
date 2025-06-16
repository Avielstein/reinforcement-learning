"""Data manager for WaterWorld training simulation and metrics."""

import time
import random
import math
from typing import Dict, Any, List
from collections import deque

from environment import WaterWorld
from config import EnvironmentConfig, AgentConfig

class DataManager:
    """Manages training data and simulation for research interface."""
    
    def __init__(self):
        # Initialize environment
        self.env_config = EnvironmentConfig()
        self.agent_config = AgentConfig()
        self.environment = WaterWorld(self.env_config)
        
        # Training state
        self.is_training = False
        self.episode = 0
        self.total_steps = 0
        self.episode_steps = 0
        
        # Metrics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.recent_losses = deque(maxlen=50)
        
        # Mock DQN parameters (adjustable via UI)
        self.learning_rate = self.agent_config.LEARNING_RATE
        self.epsilon = self.agent_config.EPSILON_START
        self.epsilon_decay = self.agent_config.EPSILON_DECAY
        self.target_update_freq = self.agent_config.TARGET_UPDATE_FREQUENCY
        self.batch_size = self.agent_config.BATCH_SIZE
        self.gamma = self.agent_config.GAMMA
        
        # Current episode tracking
        self.current_episode_reward = 0.0
        self.steps_since_target_update = 0
        
        # Initialize environment
        self.reset_episode()
    
    def start_training(self):
        """Start training simulation."""
        self.is_training = True
        print("🚀 Training started")
    
    def pause_training(self):
        """Pause training simulation."""
        self.is_training = False
        print("⏸️ Training paused")
    
    def reset_training(self):
        """Reset training to initial state."""
        self.is_training = False
        self.episode = 0
        self.total_steps = 0
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.steps_since_target_update = 0
        
        # Clear metrics
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.recent_losses.clear()
        
        # Reset epsilon
        self.epsilon = self.agent_config.EPSILON_START
        
        # Reset environment
        self.reset_episode()
        print("🔄 Training reset")
    
    def reset_episode(self):
        """Reset current episode."""
        self.environment.reset()
        self.episode_steps = 0
        self.current_episode_reward = 0.0
    
    def step(self) -> Dict[str, Any]:
        """Execute one training step and return current state."""
        if not self.is_training:
            return self.get_current_state()
        
        # Generate mock action (random with decreasing epsilon)
        if random.random() < self.epsilon:
            # Random action
            action = (random.uniform(-1, 1), random.uniform(-1, 1))
        else:
            # Mock "learned" action - move toward good items, away from bad items
            action = self._generate_mock_policy_action()
        
        # Execute environment step
        observation, reward, done, info = self.environment.step(action)
        
        # Update tracking
        self.total_steps += 1
        self.episode_steps += 1
        self.current_episode_reward += reward
        self.steps_since_target_update += 1
        
        # Mock training loss (decreases over time)
        if self.total_steps % 4 == 0:  # Train every 4 steps
            mock_loss = max(0.001, 0.1 * math.exp(-self.total_steps / 1000))
            self.recent_losses.append(mock_loss)
        
        # Update epsilon
        if self.total_steps % 10 == 0:
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        
        # Target network update
        if self.steps_since_target_update >= self.target_update_freq:
            self.steps_since_target_update = 0
        
        # Handle episode end
        if done:
            self.episode += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.episode_steps)
            self.reset_episode()
        
        return self.get_current_state()
    
    def _generate_mock_policy_action(self) -> tuple:
        """Generate mock policy action that improves over time."""
        # Get environment state
        state = self.environment.get_state_dict()
        agent = state['agent']
        good_items = state['good_items']
        bad_items = state['bad_items']
        
        # Simple heuristic: move toward nearest good item, away from bad items
        target_x, target_y = agent['x'], agent['y']
        
        # Find nearest good item
        min_good_dist = float('inf')
        for item in good_items:
            if item['active']:
                dist = math.sqrt((agent['x'] - item['x'])**2 + (agent['y'] - item['y'])**2)
                if dist < min_good_dist:
                    min_good_dist = dist
                    target_x, target_y = item['x'], item['y']
        
        # Avoid bad items
        for item in bad_items:
            if item['active']:
                dist = math.sqrt((agent['x'] - item['x'])**2 + (agent['y'] - item['y'])**2)
                if dist < 50:  # Avoidance radius
                    # Move away from bad item
                    avoid_x = agent['x'] - item['x']
                    avoid_y = agent['y'] - item['y']
                    target_x = agent['x'] + avoid_x * 2
                    target_y = agent['y'] + avoid_y * 2
        
        # Calculate action
        dx = target_x - agent['x']
        dy = target_y - agent['y']
        
        # Normalize
        magnitude = math.sqrt(dx*dx + dy*dy)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        # Add some noise for exploration
        noise_factor = self.epsilon * 0.5
        dx += random.uniform(-noise_factor, noise_factor)
        dy += random.uniform(-noise_factor, noise_factor)
        
        return (dx, dy)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current training state for UI updates."""
        # Get environment state
        env_state = self.environment.get_state_dict()
        
        # Calculate metrics
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
        avg_length = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0
        current_loss = self.recent_losses[-1] if self.recent_losses else 0.0
        
        return {
            # Training metrics
            'episode': self.episode,
            'total_steps': self.total_steps,
            'episode_steps': self.episode_steps,
            'episode_reward': self.current_episode_reward,
            'avg_reward': avg_reward,
            'avg_episode_length': avg_length,
            
            # Algorithm metrics
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'loss': current_loss,
            'steps_since_target_update': self.steps_since_target_update,
            
            # Environment state
            'environment': env_state,
            
            # Training status
            'is_training': self.is_training,
            
            # Performance history
            'reward_history': list(self.episode_rewards)[-20:],  # Last 20 episodes
            'loss_history': list(self.recent_losses)[-20:]       # Last 20 losses
        }
    
    def update_parameter(self, parameter: str, value: float):
        """Update training parameter."""
        if parameter == 'learning_rate':
            self.learning_rate = value
        elif parameter == 'epsilon_decay':
            self.epsilon_decay = value
        elif parameter == 'target_update_freq':
            self.target_update_freq = int(value)
        elif parameter == 'batch_size':
            self.batch_size = int(value)
        elif parameter == 'gamma':
            self.gamma = value
        
        print(f"📊 Parameter updated: {parameter} = {value}")
    
    def save_model(self):
        """Mock model saving."""
        print(f"💾 Model saved at episode {self.episode}")
        return f"model_episode_{self.episode}.pt"

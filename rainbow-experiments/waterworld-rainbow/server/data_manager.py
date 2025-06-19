"""Data manager for WaterWorld training simulation and metrics."""

import os
import time
import random
import math
import glob
from typing import Dict, Any, List, Tuple
from collections import deque
from datetime import datetime

from environment import WaterWorld
from agent.double_dqn import DoubleDQN
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
        
        # Training mode (mock vs real DQN)
        self.use_real_dqn = False
        self.dqn_agent = None
        
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
        
        if self.use_real_dqn and self.dqn_agent is not None:
            return self._step_real_dqn()
        else:
            return self._step_mock_training()
    
    def _step_real_dqn(self) -> Dict[str, Any]:
        """Execute one step with real DQN agent."""
        # Get current observation
        observation = self.environment._get_observation()
        
        # Get action from DQN agent
        action_idx = self.dqn_agent.get_action(observation, training=True)
        movement = self._action_to_movement(action_idx)
        
        # Execute environment step
        next_observation, reward, done, info = self.environment.step(movement)
        
        # Store experience in DQN agent
        self.dqn_agent.store_experience(observation, action_idx, reward, next_observation, done)
        
        # Train the agent
        train_metrics = self.dqn_agent.train_step()
        if train_metrics and 'loss' in train_metrics:
            self.recent_losses.append(train_metrics['loss'])
        
        # Update tracking
        self.total_steps += 1
        self.episode_steps += 1
        self.current_episode_reward += reward
        
        # Update epsilon and other metrics from DQN agent
        self.epsilon = self.dqn_agent.epsilon
        self.steps_since_target_update = self.total_steps % self.target_update_freq
        
        # Handle episode end
        if done:
            self.episode += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.episode_steps)
            self.dqn_agent.episode_rewards.append(self.current_episode_reward)
            self.dqn_agent.episodes = self.episode
            self.reset_episode()
        
        return self.get_current_state()
    
    def _step_mock_training(self) -> Dict[str, Any]:
        """Execute one step with mock training."""
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
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available trained models."""
        models = []
        
        # Search for models in the models directory
        model_patterns = [
            "models/**/*.pt",
            "models/*.pt",
            "*.pt"
        ]
        
        for pattern in model_patterns:
            for model_path in glob.glob(pattern, recursive=True):
                if os.path.isfile(model_path):
                    # Get file info
                    stat = os.stat(model_path)
                    size_mb = stat.st_size / (1024 * 1024)
                    modified_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    # Extract model name and info
                    filename = os.path.basename(model_path)
                    dirname = os.path.dirname(model_path)
                    
                    models.append({
                        'path': model_path,
                        'filename': filename,
                        'directory': dirname,
                        'size_mb': round(size_mb, 2),
                        'modified': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'modified_timestamp': stat.st_mtime
                    })
        
        # Remove duplicates and sort by modification time (newest first)
        seen_paths = set()
        unique_models = []
        for model in models:
            if model['path'] not in seen_paths:
                seen_paths.add(model['path'])
                unique_models.append(model)
        
        unique_models.sort(key=lambda x: x['modified_timestamp'], reverse=True)
        
        print(f"📁 Found {len(unique_models)} available models")
        return unique_models
    
    def load_model(self, model_path: str) -> Tuple[bool, str]:
        """Load a trained model."""
        try:
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
            
            # Initialize real DQN agent if not already done
            if not hasattr(self, 'dqn_agent') or self.dqn_agent is None:
                self._initialize_real_dqn()
            
            # Load the model
            self.dqn_agent.load(model_path)
            
            # Update training mode to use real DQN
            self.use_real_dqn = True
            
            # Reset training state to start fresh with loaded model
            self.episode = self.dqn_agent.episodes
            self.total_steps = self.dqn_agent.steps
            self.epsilon = self.dqn_agent.epsilon
            
            # Update metrics from loaded model
            if hasattr(self.dqn_agent, 'episode_rewards') and self.dqn_agent.episode_rewards:
                self.episode_rewards.extend(self.dqn_agent.episode_rewards[-50:])  # Last 50 episodes
            
            print(f"✅ Model loaded successfully: {model_path}")
            print(f"📊 Model stats - Episodes: {self.episode}, Steps: {self.total_steps}, Epsilon: {self.epsilon:.3f}")
            
            return True, f"Model loaded successfully from {os.path.basename(model_path)}"
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False, f"Error loading model: {str(e)}"
    
    def _initialize_real_dqn(self):
        """Initialize real DQN agent."""
        state_dim = self.environment.get_observation_dim()
        action_dim = 8  # 8 directional actions
        
        self.dqn_agent = DoubleDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            epsilon_start=self.agent_config.EPSILON_START,
            epsilon_end=self.agent_config.EPSILON_END,
            epsilon_decay=self.epsilon_decay,
            target_update_freq=self.target_update_freq,
            batch_size=self.batch_size,
            buffer_size=self.agent_config.REPLAY_BUFFER_SIZE,
            hidden_dims=self.agent_config.HIDDEN_LAYERS
        )
        
        # Action mapping (8 directions)
        self.action_map = [
            (0, 1),    # North
            (1, 1),    # Northeast  
            (1, 0),    # East
            (1, -1),   # Southeast
            (0, -1),   # South
            (-1, -1),  # Southwest
            (-1, 0),   # West
            (-1, 1)    # Northwest
        ]
        
        print("🧠 Real DQN agent initialized")
    
    def set_real_training_mode(self, use_real_dqn: bool):
        """Toggle between mock and real DQN training."""
        self.use_real_dqn = use_real_dqn
        
        if use_real_dqn and (not hasattr(self, 'dqn_agent') or self.dqn_agent is None):
            self._initialize_real_dqn()
        
        mode = "Real DQN" if use_real_dqn else "Mock"
        print(f"🔄 Training mode set to: {mode}")
    
    def _action_to_movement(self, action: int) -> tuple:
        """Convert discrete action to movement direction."""
        if hasattr(self, 'action_map'):
            return self.action_map[action]
        else:
            # Fallback action mapping
            action_map = [
                (0, 1), (1, 1), (1, 0), (1, -1),
                (0, -1), (-1, -1), (-1, 0), (-1, 1)
            ]
            return action_map[action]

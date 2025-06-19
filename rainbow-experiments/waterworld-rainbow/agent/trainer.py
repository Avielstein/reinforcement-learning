"""Training manager for RAINBOW DQN agent."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import time
import os
from datetime import datetime

from agent.rainbow import RainbowDQN
from config import AgentConfig, EnvironmentConfig
from environment.waterworld import WaterWorld

class RainbowTrainer:
    """Training manager for RAINBOW DQN agent."""
    
    def __init__(
        self,
        agent_config: AgentConfig = None,
        env_config: EnvironmentConfig = None,
        save_dir: str = "models"
    ):
        """
        Initialize trainer.
        
        Args:
            agent_config: Agent configuration
            env_config: Environment configuration
            save_dir: Directory to save models
        """
        self.agent_config = agent_config or AgentConfig()
        self.env_config = env_config or EnvironmentConfig()
        self.save_dir = save_dir
        
        # Create environment
        self.env = WaterWorld(self.env_config)
        
        # Set observation dimension based on environment
        self.agent_config.OBSERVATION_DIM = self.env.get_observation_dim()
        
        # Create RAINBOW agent
        self.agent = RainbowDQN(
            state_dim=self.agent_config.OBSERVATION_DIM,
            action_dim=self.agent_config.ACTION_DIM,
            learning_rate=self.agent_config.LEARNING_RATE,
            gamma=self.agent_config.GAMMA,
            epsilon_start=self.agent_config.EPSILON_START,
            epsilon_end=self.agent_config.EPSILON_END,
            epsilon_decay=self.agent_config.EPSILON_DECAY,
            target_update_freq=self.agent_config.TARGET_UPDATE_FREQUENCY,
            batch_size=self.agent_config.BATCH_SIZE,
            buffer_size=self.agent_config.REPLAY_BUFFER_SIZE,
            hidden_dims=self.agent_config.HIDDEN_LAYERS,
            n_step=self.agent_config.N_STEP,
            v_min=self.agent_config.V_MIN,
            v_max=self.agent_config.V_MAX,
            n_atoms=self.agent_config.N_ATOMS,
            noisy_std=self.agent_config.NOISY_STD
        )
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.training_start_time = None
        
        # Sample efficiency tracking
        self.best_avg_reward = -float('inf')
        self.episodes_since_improvement = 0
        self.convergence_detected = False
        self.performance_history = []
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(
        self,
        num_episodes: int = 1000,
        max_steps_per_episode: int = None,
        save_frequency: int = 100,
        eval_frequency: int = 50,
        log_frequency: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the RAINBOW agent.
        
        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
            save_frequency: Save model every N episodes
            eval_frequency: Evaluate agent every N episodes
            verbose: Print training progress
            
        Returns:
            Training statistics
        """
        if max_steps_per_episode is None:
            max_steps_per_episode = self.env_config.MAX_EPISODE_STEPS
        
        self.training_start_time = time.time()
        
        for episode in range(num_episodes):
            episode_reward = 0
            episode_steps = 0
            
            # Reset environment
            state = self.env.reset()
            
            for step in range(max_steps_per_episode):
                # Get action (RAINBOW uses noisy networks for exploration)
                action = self.agent.get_action(state, training=True)
                
                # Convert discrete action to continuous movement
                action_continuous = self._discrete_to_continuous_action(action)
                
                # Take step
                next_state, reward, done, info = self.env.step(action_continuous)
                
                # Store experience (with n-step learning)
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                if step % self.agent_config.TRAIN_FREQUENCY == 0:
                    train_info = self.agent.train_step()
                    if train_info:
                        self.losses.append(train_info['loss'])
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            # Store episode stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.agent.episode_rewards.append(episode_reward)
            self.agent.episodes += 1
            
            # Check for performance improvement and early stopping
            if len(self.episode_rewards) >= self.agent_config.PERFORMANCE_WINDOW:
                current_avg = np.mean(self.episode_rewards[-self.agent_config.PERFORMANCE_WINDOW:])
                self.performance_history.append(current_avg)
                
                # Check if this is a new best performance
                if current_avg > self.best_avg_reward + self.agent_config.CONVERGENCE_THRESHOLD:
                    improvement = current_avg - self.best_avg_reward
                    self.best_avg_reward = current_avg
                    self.episodes_since_improvement = 0
                    
                    # Save new best model immediately
                    best_filepath = os.path.join(self.save_dir, f"best_rainbow_ep{episode+1}_reward{current_avg:.2f}.pt")
                    self.agent.save(best_filepath)
                    
                    if verbose:
                        print(f"🎉 NEW BEST! Episode {episode + 1}: Avg Reward {current_avg:.2f} (+{improvement:.2f}) - Model saved!")
                else:
                    self.episodes_since_improvement += 1
                
                # Early stopping check
                if self.episodes_since_improvement >= self.agent_config.EARLY_STOPPING_PATIENCE:
                    if verbose:
                        print(f"🛑 Early stopping triggered after {episode + 1} episodes")
                        print(f"No improvement for {self.agent_config.EARLY_STOPPING_PATIENCE} episodes")
                        print(f"Best average reward: {self.best_avg_reward:.2f}")
                    self.convergence_detected = True
                    break
            
            # Print progress
            if verbose and (episode + 1) % log_frequency == 0:
                avg_reward = np.mean(self.episode_rewards[-log_frequency:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                episodes_since = self.episodes_since_improvement if len(self.episode_rewards) >= self.agent_config.PERFORMANCE_WINDOW else 0
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Best: {self.best_avg_reward:.2f}, "
                      f"No improve: {episodes_since}, "
                      f"Buffer: {len(self.agent.replay_buffer)}, "
                      f"Loss: {avg_loss:.4f}")
            
            # Evaluate agent and save best models
            if (episode + 1) % eval_frequency == 0:
                eval_reward = self.evaluate(num_episodes=5, verbose=False)
                
                # Check if this is a new best evaluation performance
                if not hasattr(self, 'best_eval_reward') or eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    
                    # Save new best evaluation model
                    eval_filepath = os.path.join(self.save_dir, f"best_eval_ep{episode+1}_reward{eval_reward:.2f}.pt")
                    self.agent.save(eval_filepath)
                    
                    if verbose:
                        print(f"🏆 NEW BEST EVAL! Episode {episode + 1}: Eval Reward {eval_reward:.2f} - Model saved!")
                else:
                    if verbose:
                        print(f"📊 Evaluation reward: {eval_reward:.2f} (Best: {self.best_eval_reward:.2f})")
        
        # Save final model
        self.save_model("final_model")
        
        return self.get_training_stats()
    
    def evaluate(self, num_episodes: int = 10, verbose: bool = True) -> float:
        """
        Evaluate the agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            verbose: Print evaluation progress
            
        Returns:
            Average reward over evaluation episodes
        """
        eval_rewards = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            state = self.env.reset()
            
            for step in range(self.env_config.MAX_EPISODE_STEPS):
                # Get action (no exploration noise for evaluation)
                action = self.agent.get_action(state, training=False)
                action_continuous = self._discrete_to_continuous_action(action)
                
                # Take step
                next_state, reward, done, info = self.env.step(action_continuous)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            
            if verbose:
                print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        avg_reward = np.mean(eval_rewards)
        if verbose:
            print(f"Average Evaluation Reward: {avg_reward:.2f}")
        
        return avg_reward
    
    def _discrete_to_continuous_action(self, discrete_action: int) -> tuple:
        """Convert discrete action to continuous movement."""
        # Map discrete actions to movement directions
        action_map = {
            0: (1.0, 0.0),   # Right
            1: (-1.0, 0.0),  # Left
            2: (0.0, 1.0),   # Up
            3: (0.0, -1.0),  # Down
            4: (0.7, 0.7),   # Up-Right
            5: (-0.7, 0.7),  # Up-Left
            6: (0.7, -0.7),  # Down-Right
            7: (-0.7, -0.7), # Down-Left
        }
        
        # Default to 8 actions, but handle case where ACTION_DIM is 2
        if self.agent_config.ACTION_DIM == 2:
            # Simple 2-action case: left/right or up/down
            if discrete_action == 0:
                return (1.0, 0.0)
            else:
                return (-1.0, 0.0)
        
        return action_map.get(discrete_action, (0.0, 0.0))
    
    def save_model(self, filename: str):
        """Save the trained model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.save_dir, f"waterworld_rainbow_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        filepath = os.path.join(model_dir, f"{filename}.pt")
        self.agent.save(filepath)
        
        # Also save as best model if this is the best performance
        if self.episode_rewards and len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            if not hasattr(self, 'best_avg_reward') or recent_avg > self.best_avg_reward:
                self.best_avg_reward = recent_avg
                best_filepath = os.path.join(model_dir, "best_waterworld_rainbow.pt")
                self.agent.save(best_filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.agent.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'total_steps': self.agent.steps,
            'total_episodes': len(self.episode_rewards),
            'training_time': training_time,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'agent_stats': self.agent.get_stats()
        }
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Moving average of rewards
        if len(self.episode_rewards) >= 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title('Moving Average Reward (10 episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
        
        # Losses
        if self.losses:
            axes[1, 0].plot(self.losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Episode lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training progress plot saved to {save_path}")
        
        plt.show()

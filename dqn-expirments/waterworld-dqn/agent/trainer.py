"""Training module for Double DQN in WaterWorld."""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import torch

from agent.double_dqn import DoubleDQN
from environment import WaterWorld
from config import EnvironmentConfig, AgentConfig

class DQNTrainer:
    """Trainer for Double DQN agent in WaterWorld environment."""
    
    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        save_dir: str = "models",
        log_interval: int = 100,
        save_interval: int = 1000,
        eval_interval: int = 500,
        eval_episodes: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            env_config: Environment configuration
            agent_config: Agent configuration
            save_dir: Directory to save models
            log_interval: Interval for logging
            save_interval: Interval for saving models
            eval_interval: Interval for evaluation
            eval_episodes: Number of episodes for evaluation
        """
        self.env_config = env_config or EnvironmentConfig()
        self.agent_config = agent_config or AgentConfig()
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize environment
        self.env = WaterWorld(self.env_config)
        
        # Initialize agent with expanded discrete action space
        state_dim = self.env.get_observation_dim()
        action_dim = 25  # 5x5 grid of movement directions for fine control
        
        self.agent = DoubleDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=self.agent_config.LEARNING_RATE,
            gamma=self.agent_config.GAMMA,
            epsilon_start=self.agent_config.EPSILON_START,
            epsilon_end=self.agent_config.EPSILON_END,
            epsilon_decay=self.agent_config.EPSILON_DECAY,
            target_update_freq=self.agent_config.TARGET_UPDATE_FREQUENCY,
            batch_size=self.agent_config.BATCH_SIZE,
            buffer_size=self.agent_config.REPLAY_BUFFER_SIZE,
            hidden_dims=self.agent_config.HIDDEN_LAYERS
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.eval_rewards = []
        
        # Create fine-grained action mapping (5x5 grid)
        self.action_map = []
        for dx in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for dy in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                self.action_map.append((dx, dy))
    
    def action_to_movement(self, action: int) -> tuple:
        """Convert discrete action to movement direction."""
        return self.action_map[action]
    
    def train(
        self,
        num_episodes: int = 2000,
        max_steps_per_episode: int = 1000,
        warmup_episodes: int = 100
    ) -> Dict[str, List]:
        """
        Train the Double DQN agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            warmup_episodes: Episodes before training starts
            
        Returns:
            Training history
        """
        print(f"🚀 Starting Double DQN training for {num_episodes} episodes")
        print(f"📊 Environment: {self.env_config.WORLD_WIDTH}x{self.env_config.WORLD_HEIGHT}")
        print(f"🧠 State dim: {self.agent.state_dim}, Action dim: {self.agent.action_dim}")
        print(f"💾 Models will be saved to: {self.save_dir}")
        print("-" * 60)
        
        start_time = time.time()
        best_eval_reward = -float('inf')
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                # Get action
                action = self.agent.get_action(state, training=True)
                movement = self.action_to_movement(action)
                
                # Take step
                next_state, reward, done, info = self.env.step(movement)
                
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent (after warmup)
                if episode >= warmup_episodes:
                    train_metrics = self.agent.train_step()
                    if train_metrics:
                        self.training_losses.append(train_metrics['loss'])
                
                # Decay epsilon every step (not just during training)
                if self.agent.epsilon > self.agent.epsilon_end:
                    self.agent.epsilon *= self.agent.epsilon_decay
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.agent.episode_rewards.append(episode_reward)
            self.agent.episodes += 1
            
            # Logging
            if (episode + 1) % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.log_interval:])
                avg_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0
                
                elapsed_time = time.time() - start_time
                episodes_per_sec = (episode + 1) / elapsed_time
                
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:6.1f} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(self.agent.replay_buffer):5d} | "
                      f"EPS: {episodes_per_sec:.1f}")
            
            # Evaluation
            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                
                print(f"🎯 Evaluation after {episode + 1} episodes: {eval_reward:.2f}")
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_model_path = os.path.join(self.save_dir, "best_waterworld_dqn.pt")
                    self.agent.save(best_model_path)
                    print(f"💾 New best model saved: {eval_reward:.2f}")
            
            # Save checkpoint
            if (episode + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode + 1}.pt")
                self.agent.save(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Final save
        final_model_path = os.path.join(self.save_dir, "final_waterworld_dqn.pt")
        self.agent.save(final_model_path)
        
        # Save training plots
        self.save_training_plots()
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time:.1f} seconds")
        print(f"📈 Best evaluation reward: {best_eval_reward:.2f}")
        print(f"💾 Final model saved: {final_model_path}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'eval_rewards': self.eval_rewards
        }
    
    def evaluate(self) -> float:
        """Evaluate agent performance."""
        total_reward = 0
        
        for _ in range(self.eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(1000):  # Max steps for evaluation
                action = self.agent.get_action(state, training=False)
                movement = self.action_to_movement(action)
                state, reward, done, _ = self.env.step(movement)
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / self.eval_episodes
    
    def save_training_plots(self):
        """Save training progress plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Moving average of rewards
        if len(self.episode_rewards) > 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title('Moving Average Reward (100 episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
        
        # Training loss
        if self.training_losses:
            axes[1, 0].plot(self.training_losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Evaluation rewards
        if self.eval_rewards:
            eval_episodes = np.arange(len(self.eval_rewards)) * self.eval_interval + self.eval_interval
            axes[1, 1].plot(eval_episodes, self.eval_rewards, 'o-')
            axes[1, 1].set_title('Evaluation Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved: {plot_path}")
    
    def load_and_test(self, model_path: str, num_episodes: int = 10):
        """Load trained model and test performance."""
        print(f"🔄 Loading model from: {model_path}")
        self.agent.load(model_path)
        
        print(f"🧪 Testing for {num_episodes} episodes...")
        total_reward = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(1000):
                action = self.agent.get_action(state, training=False)
                movement = self.action_to_movement(action)
                state, reward, done, info = self.env.step(movement)
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            total_reward += episode_reward
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        
        avg_reward = total_reward / num_episodes
        print(f"\n📊 Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        
        return avg_reward

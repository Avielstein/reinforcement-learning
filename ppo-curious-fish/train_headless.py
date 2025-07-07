#!/usr/bin/env python3
"""
Headless training script for PPO + Curiosity Fish.

Trains the fish without web interface for maximum performance.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import FishWaterworld
from agent import PPOCuriousAgent

def train_fish(total_steps=100000, save_interval=10000, log_interval=1000):
    """
    Train the curious fish in headless mode.
    
    Args:
        total_steps: Total training steps
        save_interval: Steps between model saves
        log_interval: Steps between progress logs
    """
    print("🐟 PPO + Curiosity Fish: Headless Training")
    print("=" * 50)
    
    # Create environment and agent
    env = FishWaterworld()
    agent = PPOCuriousAgent()
    
    print(f"Environment: {env.width}x{env.height} with {env.num_food} food + {env.num_poison} poison")
    print(f"Agent: PPO + ICM with {agent.state_dim}D state, {agent.action_dim}D action")
    print(f"Training for {total_steps:,} steps...")
    print()
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    curiosity_rewards = []
    
    # Current episode tracking
    current_episode_reward = 0.0
    current_episode_length = 0
    episode_count = 0
    
    # Reset environment
    state = env.reset()
    
    # Training loop
    start_time = time.time()
    
    for step in range(total_steps):
        # Get action from agent
        action, log_prob, value = agent.get_action(state, training=True)
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Update metrics
        current_episode_reward += reward
        current_episode_length += 1
        
        # Update agent if memory is full
        if agent.memory.is_full():
            metrics = agent.update()
            if metrics:
                training_losses.append({
                    'step': step,
                    'actor_loss': metrics['actor_loss'],
                    'critic_loss': metrics['critic_loss'],
                    'curiosity_loss': metrics.get('icm_loss', 0.0),
                    'prediction_error': metrics.get('prediction_error', 0.0)
                })
        
        # Track curiosity rewards
        if hasattr(agent, 'curiosity_module'):
            import torch
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
            intrinsic_reward = agent.curiosity_module.compute_intrinsic_reward(
                state_tensor, action_tensor, next_state_tensor
            ).item()
            curiosity_rewards.append(intrinsic_reward)
        
        # Episode end (in this environment, episodes don't naturally end, so we'll create artificial episodes)
        if current_episode_length >= 2000:  # 2000 steps per episode
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            episode_count += 1
            
            # Reset episode tracking
            current_episode_reward = 0.0
            current_episode_length = 0
            
            # Reset environment occasionally for variety
            if episode_count % 5 == 0:
                state = env.reset()
            else:
                state = next_state
        else:
            state = next_state
        
        # Logging
        if (step + 1) % log_interval == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed_time
            
            # Recent performance
            recent_rewards = episode_rewards[-10:] if episode_rewards else [0]
            recent_curiosity = curiosity_rewards[-1000:] if curiosity_rewards else [0]
            recent_losses = training_losses[-10:] if training_losses else []
            
            print(f"Step {step+1:,}/{total_steps:,} ({(step+1)/total_steps*100:.1f}%)")
            print(f"  Time: {elapsed_time:.1f}s | Speed: {steps_per_sec:.1f} steps/s")
            print(f"  Episodes: {episode_count} | Avg Reward (last 10): {np.mean(recent_rewards):.2f}")
            print(f"  Avg Curiosity: {np.mean(recent_curiosity):.4f}")
            if recent_losses:
                print(f"  Actor Loss: {np.mean([l['actor_loss'] for l in recent_losses]):.4f}")
                print(f"  Critic Loss: {np.mean([l['critic_loss'] for l in recent_losses]):.4f}")
            print(f"  Memory: {len(agent.memory)}/{agent.memory.buffer_size}")
            print()
        
        # Save model
        if (step + 1) % save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/curious_fish_step_{step+1}_{timestamp}.pt"
            os.makedirs("models", exist_ok=True)
            agent.save(model_path)
            print(f"💾 Model saved: {model_path}")
            print()
    
    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"models/curious_fish_final_{timestamp}.pt"
    agent.save(final_model_path)
    
    # Training summary
    total_time = time.time() - start_time
    print("🎉 Training Complete!")
    print("=" * 50)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Total episodes: {episode_count}")
    print(f"Final model: {final_model_path}")
    print(f"Average reward (last 20 episodes): {np.mean(episode_rewards[-20:]) if episode_rewards else 0:.2f}")
    print(f"Training updates: {len(training_losses)}")
    
    # Plot training curves
    if episode_rewards and training_losses:
        plot_training_results(episode_rewards, training_losses, curiosity_rewards)
    
    return final_model_path, {
        'episode_rewards': episode_rewards,
        'training_losses': training_losses,
        'curiosity_rewards': curiosity_rewards,
        'total_time': total_time,
        'total_episodes': episode_count
    }

def plot_training_results(episode_rewards, training_losses, curiosity_rewards):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Training losses
    if training_losses:
        steps = [l['step'] for l in training_losses]
        actor_losses = [l['actor_loss'] for l in training_losses]
        critic_losses = [l['critic_loss'] for l in training_losses]
        
        axes[0, 1].plot(steps, actor_losses, label='Actor Loss')
        axes[0, 1].plot(steps, critic_losses, label='Critic Loss')
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Curiosity rewards
    if curiosity_rewards:
        # Moving average for smoother curve
        window = 100
        if len(curiosity_rewards) > window:
            smoothed = np.convolve(curiosity_rewards, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(smoothed)
        else:
            axes[1, 0].plot(curiosity_rewards)
        axes[1, 0].set_title('Curiosity Rewards (Smoothed)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Intrinsic Reward')
        axes[1, 0].grid(True)
    
    # Reward distribution
    if episode_rewards:
        axes[1, 1].hist(episode_rewards, bins=20, alpha=0.7)
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Episode Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"training_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training plots saved: {plot_path}")
    
    # Show plot if possible
    try:
        plt.show()
    except:
        pass  # Headless environment might not support display

def test_trained_model(model_path, test_steps=5000):
    """Test a trained model."""
    print(f"\n🧪 Testing trained model: {model_path}")
    print("-" * 50)
    
    # Create environment and agent
    env = FishWaterworld()
    agent = PPOCuriousAgent()
    
    # Load trained model
    try:
        agent.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test the agent
    state = env.reset()
    total_reward = 0.0
    food_eaten = 0
    poison_eaten = 0
    
    for step in range(test_steps):
        # Get action (no training)
        action, _, _ = agent.get_action(state, training=False)
        
        # Step environment
        state, reward, done, info = env.step(action)
        
        total_reward += reward
        food_eaten += info.get('food_eaten', 0)
        poison_eaten += info.get('poison_eaten', 0)
        
        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{test_steps}: Reward={total_reward:.2f}, Food={food_eaten}, Poison={poison_eaten}")
    
    print(f"\n📊 Test Results:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Food eaten: {food_eaten}")
    print(f"Poison eaten: {poison_eaten}")
    print(f"Net score: {food_eaten - poison_eaten}")
    print(f"Average reward per step: {total_reward/test_steps:.4f}")

if __name__ == "__main__":
    # Train the fish
    model_path, results = train_fish(
        total_steps=50000,  # 50k steps should be enough to see good learning
        save_interval=10000,
        log_interval=2000
    )
    
    # Test the trained model
    test_trained_model(model_path)
    
    print(f"\n🎯 Training complete! Use this model in the web interface:")
    print(f"   {model_path}")

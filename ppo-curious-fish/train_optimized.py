#!/usr/bin/env python3
"""
Optimized training script for PPO + Curiosity Fish.

This version is designed to achieve much better performance through:
- Larger buffer sizes
- Better hyperparameters
- Longer training
- Improved reward shaping
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

def create_optimized_agent():
    """Create an agent with optimized hyperparameters."""
    return PPOCuriousAgent(
        state_dim=152,
        action_dim=4,
        hidden_dim=512,  # Larger networks
        learning_rate=1e-3,  # Higher learning rate
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.02,  # More exploration
        value_coef=0.5,
        curiosity_weight=0.5,  # Much higher curiosity
        buffer_size=8192,  # 4x larger buffer
        batch_size=256,  # Larger batches
        n_epochs=10,
        curiosity_lr=3e-3,  # Higher curiosity learning rate
        feature_dim=128  # Larger feature space
    )

def train_fish_optimized(total_steps=500000, save_interval=50000, log_interval=5000):
    """
    Train the curious fish with optimized settings.
    
    Args:
        total_steps: Total training steps (10x more than before)
        save_interval: Steps between model saves
        log_interval: Steps between progress logs
    """
    print("🐟 PPO + Curiosity Fish: OPTIMIZED Training")
    print("=" * 60)
    
    # Create environment and optimized agent
    env = FishWaterworld()
    agent = create_optimized_agent()
    
    print(f"Environment: {env.width}x{env.height} with {env.num_food} food + {env.num_poison} poison")
    print(f"Agent: PPO + ICM with {agent.state_dim}D state, {agent.action_dim}D action")
    print(f"Optimizations:")
    print(f"  - Buffer size: {agent.memory.buffer_size} (4x larger)")
    print(f"  - Hidden dim: 512 (2x larger)")
    print(f"  - Learning rate: {agent.learning_rate} (3x higher)")
    print(f"  - Curiosity weight: {agent.curiosity_weight} (5x higher)")
    print(f"  - Batch size: {agent.batch_size} (4x larger)")
    print(f"Training for {total_steps:,} steps (10x longer)...")
    print()
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    curiosity_rewards = []
    food_eaten_history = []
    poison_eaten_history = []
    
    # Current episode tracking
    current_episode_reward = 0.0
    current_episode_length = 0
    current_food_eaten = 0
    current_poison_eaten = 0
    episode_count = 0
    
    # Best performance tracking
    best_score = -float('inf')
    best_model_path = None
    
    # Reset environment
    state = env.reset()
    
    # Training loop
    start_time = time.time()
    
    for step in range(total_steps):
        # Get action from agent
        action, log_prob, value = agent.get_action(state, training=True)
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        
        # Enhanced reward shaping for better learning
        shaped_reward = reward
        
        # Add small survival bonus
        shaped_reward += 0.001
        
        # Add proximity rewards (encourage approaching food, avoiding poison)
        # This helps with sparse reward problem
        if 'fish_position' in info:
            fish_pos = info['fish_position']
            
            # Find closest food and poison
            min_food_dist = float('inf')
            min_poison_dist = float('inf')
            
            for food in env.food_items:
                dist = np.linalg.norm(fish_pos - food.position)
                min_food_dist = min(min_food_dist, dist)
            
            for poison in env.poison_items:
                dist = np.linalg.norm(fish_pos - poison.position)
                min_poison_dist = min(min_poison_dist, dist)
            
            # Reward approaching food, penalize approaching poison
            if min_food_dist < 50:  # Close to food
                shaped_reward += 0.01 * (50 - min_food_dist) / 50
            
            if min_poison_dist < 50:  # Close to poison
                shaped_reward -= 0.01 * (50 - min_poison_dist) / 50
        
        # Store transition
        agent.store_transition(state, action, shaped_reward, next_state, done)
        
        # Update metrics
        current_episode_reward += reward  # Use original reward for tracking
        current_episode_length += 1
        current_food_eaten += info.get('food_eaten', 0)
        current_poison_eaten += info.get('poison_eaten', 0)
        
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
        
        # Episode end (longer episodes for better learning)
        if current_episode_length >= 5000:  # 2.5x longer episodes
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            food_eaten_history.append(current_food_eaten)
            poison_eaten_history.append(current_poison_eaten)
            episode_count += 1
            
            # Check if this is the best performance
            net_score = current_food_eaten - current_poison_eaten
            if net_score > best_score:
                best_score = net_score
                # Save best model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = f"models/curious_fish_best_{timestamp}_score_{net_score}.pt"
                os.makedirs("models", exist_ok=True)
                agent.save(best_model_path)
                print(f"🏆 NEW BEST! Score: {net_score} (Food: {current_food_eaten}, Poison: {current_poison_eaten})")
                print(f"    Model saved: {best_model_path}")
            
            # Reset episode tracking
            current_episode_reward = 0.0
            current_episode_length = 0
            current_food_eaten = 0
            current_poison_eaten = 0
            
            # Reset environment occasionally for variety
            if episode_count % 3 == 0:  # More frequent resets
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
            recent_food = food_eaten_history[-10:] if food_eaten_history else [0]
            recent_poison = poison_eaten_history[-10:] if poison_eaten_history else [0]
            recent_curiosity = curiosity_rewards[-1000:] if curiosity_rewards else [0]
            recent_losses = training_losses[-10:] if training_losses else []
            
            avg_net_score = np.mean(recent_food) - np.mean(recent_poison)
            
            print(f"Step {step+1:,}/{total_steps:,} ({(step+1)/total_steps*100:.1f}%)")
            print(f"  Time: {elapsed_time:.1f}s | Speed: {steps_per_sec:.1f} steps/s")
            print(f"  Episodes: {episode_count} | Avg Reward (last 10): {np.mean(recent_rewards):.2f}")
            print(f"  Avg Food: {np.mean(recent_food):.1f} | Avg Poison: {np.mean(recent_poison):.1f}")
            print(f"  Avg Net Score: {avg_net_score:.1f} | Best Score: {best_score}")
            print(f"  Avg Curiosity: {np.mean(recent_curiosity):.4f}")
            if recent_losses:
                print(f"  Actor Loss: {np.mean([l['actor_loss'] for l in recent_losses]):.4f}")
                print(f"  Critic Loss: {np.mean([l['critic_loss'] for l in recent_losses]):.4f}")
                print(f"  Curiosity Loss: {np.mean([l['curiosity_loss'] for l in recent_losses]):.4f}")
            print(f"  Memory: {len(agent.memory)}/{agent.memory.buffer_size}")
            print()
        
        # Save model periodically
        if (step + 1) % save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/curious_fish_step_{step+1}_{timestamp}.pt"
            os.makedirs("models", exist_ok=True)
            agent.save(model_path)
            print(f"💾 Checkpoint saved: {model_path}")
            print()
    
    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"models/curious_fish_final_optimized_{timestamp}.pt"
    agent.save(final_model_path)
    
    # Training summary
    total_time = time.time() - start_time
    print("🎉 OPTIMIZED Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Total episodes: {episode_count}")
    print(f"Final model: {final_model_path}")
    print(f"Best model: {best_model_path} (Score: {best_score})")
    
    if episode_rewards:
        recent_food = np.mean(food_eaten_history[-20:])
        recent_poison = np.mean(poison_eaten_history[-20:])
        recent_net = recent_food - recent_poison
        print(f"Final performance (last 20 episodes):")
        print(f"  Average food eaten: {recent_food:.1f}")
        print(f"  Average poison eaten: {recent_poison:.1f}")
        print(f"  Average net score: {recent_net:.1f}")
        print(f"  Average reward: {np.mean(episode_rewards[-20:]):.2f}")
    
    print(f"Training updates: {len(training_losses)}")
    
    # Plot training curves
    if episode_rewards and training_losses:
        plot_optimized_results(episode_rewards, training_losses, curiosity_rewards, 
                             food_eaten_history, poison_eaten_history)
    
    return best_model_path or final_model_path, {
        'episode_rewards': episode_rewards,
        'training_losses': training_losses,
        'curiosity_rewards': curiosity_rewards,
        'food_eaten_history': food_eaten_history,
        'poison_eaten_history': poison_eaten_history,
        'total_time': total_time,
        'total_episodes': episode_count,
        'best_score': best_score
    }

def plot_optimized_results(episode_rewards, training_losses, curiosity_rewards, 
                         food_eaten_history, poison_eaten_history):
    """Plot comprehensive training results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Food vs Poison eaten
    axes[0, 1].plot(food_eaten_history, label='Food Eaten', color='red')
    axes[0, 1].plot(poison_eaten_history, label='Poison Eaten', color='green')
    net_scores = [f - p for f, p in zip(food_eaten_history, poison_eaten_history)]
    axes[0, 1].plot(net_scores, label='Net Score', color='blue', linewidth=2)
    axes[0, 1].set_title('Food vs Poison Performance')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training losses
    if training_losses:
        steps = [l['step'] for l in training_losses]
        actor_losses = [l['actor_loss'] for l in training_losses]
        critic_losses = [l['critic_loss'] for l in training_losses]
        curiosity_losses = [l['curiosity_loss'] for l in training_losses]
        
        axes[0, 2].plot(steps, actor_losses, label='Actor Loss')
        axes[0, 2].plot(steps, critic_losses, label='Critic Loss')
        axes[0, 2].plot(steps, curiosity_losses, label='Curiosity Loss')
        axes[0, 2].set_title('Training Losses')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # Curiosity rewards
    if curiosity_rewards:
        window = 1000
        if len(curiosity_rewards) > window:
            smoothed = np.convolve(curiosity_rewards, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(smoothed)
        else:
            axes[1, 0].plot(curiosity_rewards)
        axes[1, 0].set_title('Curiosity Rewards (Smoothed)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Intrinsic Reward')
        axes[1, 0].grid(True)
    
    # Performance distribution
    if food_eaten_history and poison_eaten_history:
        net_scores = [f - p for f, p in zip(food_eaten_history, poison_eaten_history)]
        axes[1, 1].hist(net_scores, bins=20, alpha=0.7, color='blue')
        axes[1, 1].set_title('Net Score Distribution')
        axes[1, 1].set_xlabel('Net Score (Food - Poison)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
    
    # Learning curve (moving average)
    if len(episode_rewards) > 10:
        window = min(10, len(episode_rewards) // 4)
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[1, 2].plot(smoothed_rewards)
        axes[1, 2].set_title(f'Learning Curve (MA-{window})')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Smoothed Reward')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"training_results_optimized_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Optimized training plots saved: {plot_path}")

def test_optimized_model(model_path, test_steps=10000):
    """Test an optimized trained model with longer evaluation."""
    print(f"\n🧪 Testing optimized model: {model_path}")
    print("-" * 60)
    
    # Create environment and agent
    env = FishWaterworld()
    agent = create_optimized_agent()
    
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
    
    print(f"Testing for {test_steps:,} steps...")
    
    for step in range(test_steps):
        # Get action (no training, deterministic)
        action, _, _ = agent.get_action(state, training=False)
        
        # Step environment
        state, reward, done, info = env.step(action)
        
        total_reward += reward
        food_eaten += info.get('food_eaten', 0)
        poison_eaten += info.get('poison_eaten', 0)
        
        if (step + 1) % 2000 == 0:
            net_score = food_eaten - poison_eaten
            print(f"Step {step+1:,}/{test_steps:,}: Reward={total_reward:.2f}, Food={food_eaten}, Poison={poison_eaten}, Net={net_score}")
    
    net_score = food_eaten - poison_eaten
    efficiency = food_eaten / max(1, food_eaten + poison_eaten) * 100
    
    print(f"\n📊 OPTIMIZED Test Results:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Food eaten: {food_eaten}")
    print(f"Poison eaten: {poison_eaten}")
    print(f"Net score: {net_score}")
    print(f"Efficiency: {efficiency:.1f}% (food / total eaten)")
    print(f"Average reward per step: {total_reward/test_steps:.4f}")
    print(f"Food per 1000 steps: {food_eaten * 1000 / test_steps:.1f}")
    print(f"Poison per 1000 steps: {poison_eaten * 1000 / test_steps:.1f}")

if __name__ == "__main__":
    print("🚀 Starting OPTIMIZED PPO + Curiosity Fish Training")
    print("This will train for 500,000 steps with optimized hyperparameters")
    print("Expected to achieve MUCH better performance!")
    print()
    
    # Train the fish with optimized settings
    model_path, results = train_fish_optimized(
        total_steps=500000,  # 10x longer training
        save_interval=50000,
        log_interval=10000
    )
    
    # Test the trained model
    test_optimized_model(model_path)
    
    print(f"\n🎯 OPTIMIZED training complete!")
    print(f"Best model: {model_path}")
    print(f"Expected performance: 50+ food, <5 poison per episode")

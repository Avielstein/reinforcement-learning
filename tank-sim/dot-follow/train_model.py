#!/usr/bin/env python3
"""
Dot Follow RL Training - Consolidated Python Script

This script provides a complete training pipeline for the dot follow reinforcement learning task.

Key Features:
- Primary Training: Random walk pattern (most challenging and generalizable)
- Testing: All movement patterns (circular, figure8, random_walk, zigzag, spiral)
- Model Saving: Automatic saving of best models for web interface
- Progress Tracking: Real-time visualization of training progress
- Comprehensive Evaluation: Performance analysis across all patterns

Usage:
    python train_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import time
import os
from collections import deque
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

print(f"🐟 Dot Follow RL Training Script")
print(f"Current directory: {current_dir}")
print("="*60)

# Import project modules
try:
    from dot_follow_environment import DotFollowEnv, MovingTarget
    print("✅ DotFollowEnv imported successfully")
except ImportError as e:
    print(f"❌ DotFollowEnv import failed: {e}")
    print("Make sure you're running this script from the tank-sim/dot-follow directory")
    sys.exit(1)

try:
    from dot_follow_trainer import DotFollowLearner
    print("✅ DotFollowLearner imported successfully")
except ImportError as e:
    print(f"❌ DotFollowLearner import failed: {e}")
    print("Make sure you're running this script from the tank-sim/dot-follow directory")
    sys.exit(1)

# Training Configuration
TRAINING_EPISODES = 3000
EVAL_INTERVAL = 50
SAVE_INTERVAL = 200
PATIENCE = 400  # Early stopping patience
MOVING_AVERAGE_WINDOW = 100

# Primary training pattern (most challenging for generalization)
PRIMARY_PATTERN = 'random_walk'

# Test patterns for evaluation
TEST_PATTERNS = ['circular', 'figure8', 'random_walk', 'zigzag', 'spiral']

def setup_directories():
    """Create necessary directories for saving models and plots"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    print("✅ Directories created")

def train_model():
    """Main training function"""
    print(f"\n🚀 Starting training on {PRIMARY_PATTERN} pattern...")
    print(f"Training Configuration:")
    print(f"  Episodes: {TRAINING_EPISODES}")
    print(f"  Primary pattern: {PRIMARY_PATTERN}")
    print(f"  Evaluation interval: {EVAL_INTERVAL} episodes")
    print(f"  Model checkpoints: every {SAVE_INTERVAL} episodes")
    print(f"  Early stopping patience: {PATIENCE} episodes")
    print("="*60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize learner with random walk pattern
    learner = DotFollowLearner(PRIMARY_PATTERN)

    # Training metrics tracking
    training_metrics = {
        'episodes': [],
        'rewards': [],
        'distances': [],
        'eval_rewards': [],
        'eval_episodes': [],
        'best_reward': -np.inf,
        'best_episode': 0,
        'training_time': []
    }

    # Moving averages for smoothing
    reward_ma = deque(maxlen=MOVING_AVERAGE_WINDOW)
    distance_ma = deque(maxlen=MOVING_AVERAGE_WINDOW)

    # Training loop
    start_time = time.time()
    best_eval_reward = -np.inf
    episodes_without_improvement = 0

    for episode in range(TRAINING_EPISODES):
        episode_start = time.time()
        
        # Training step
        learner.train_step()
        
        episode_time = time.time() - episode_start
        training_metrics['training_time'].append(episode_time)
        
        # Get current episode metrics
        if learner.ep_returns:
            current_reward = list(learner.ep_returns)[-1]
            reward_ma.append(current_reward)
            
            # Get performance metrics
            metrics = learner.get_performance_metrics()
            if metrics:
                distance_ma.append(metrics['mean_target_distance'])
                
                # Store metrics
                training_metrics['episodes'].append(episode)
                training_metrics['rewards'].append(np.mean(reward_ma))
                training_metrics['distances'].append(np.mean(distance_ma))
        
        # Periodic evaluation and logging
        if episode % EVAL_INTERVAL == 0 and episode > 0:
            # Quick evaluation on primary pattern
            env = DotFollowEnv(PRIMARY_PATTERN)
            obs = env.reset()
            total_reward = 0
            
            for _ in range(300):  # Evaluation episode length
                action = learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            
            training_metrics['eval_rewards'].append(total_reward)
            training_metrics['eval_episodes'].append(episode)
            
            # Check for improvement
            if total_reward > best_eval_reward:
                best_eval_reward = total_reward
                training_metrics['best_reward'] = total_reward
                training_metrics['best_episode'] = episode
                episodes_without_improvement = 0
                
                # Save best model
                learner.save_model('models/best_dot_follow_model.pt')
                print(f"🏆 New best model saved! Eval reward: {total_reward:.2f}")
            else:
                episodes_without_improvement += EVAL_INTERVAL
            
            # Progress logging
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(reward_ma) if reward_ma else 0
            avg_distance = np.mean(distance_ma) if distance_ma else 0
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Distance: {avg_distance:5.2f} | "
                  f"Eval: {total_reward:6.2f} | "
                  f"Time: {elapsed_time/60:4.1f}m | "
                  f"Best: {best_eval_reward:6.2f}@{training_metrics['best_episode']}")
        
        # Save checkpoint
        if episode % SAVE_INTERVAL == 0 and episode > 0:
            checkpoint_path = f'models/checkpoint_episode_{episode}.pt'
            learner.save_model(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Early stopping check
        if episodes_without_improvement >= PATIENCE:
            print(f"\n🛑 Early stopping triggered after {episodes_without_improvement} episodes without improvement")
            break

    total_training_time = time.time() - start_time
    print(f"\n🎯 Training completed!")
    print(f"Total time: {total_training_time/60:.1f} minutes")
    print(f"Episodes trained: {episode + 1}")
    print(f"Best evaluation reward: {best_eval_reward:.2f} at episode {training_metrics['best_episode']}")
    print(f"Average time per episode: {np.mean(training_metrics['training_time']):.3f}s")

    # Load the best model for evaluation
    learner.load_model('models/best_dot_follow_model.pt')
    print("✅ Best model loaded for evaluation")
    
    return learner, training_metrics, total_training_time

def create_training_plots(training_metrics, total_training_time):
    """Create and save training progress visualization"""
    print("\n📊 Creating training progress plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training rewards over time
    ax1 = axes[0, 0]
    if training_metrics['rewards']:
        ax1.plot(training_metrics['episodes'], training_metrics['rewards'], 'b-', linewidth=2, alpha=0.7, label='Moving Average')
        ax1.scatter(training_metrics['eval_episodes'], training_metrics['eval_rewards'], 
                    c='red', s=50, alpha=0.8, label='Evaluation', zorder=5)
        ax1.axhline(y=training_metrics['best_reward'], color='green', linestyle='--', 
                    alpha=0.7, label=f'Best: {training_metrics["best_reward"]:.2f}')
        ax1.set_title('Training Progress: Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Average distance to target
    ax2 = axes[0, 1]
    if training_metrics['distances']:
        ax2.plot(training_metrics['episodes'], training_metrics['distances'], 'r-', linewidth=2, alpha=0.7)
        ax2.set_title('Average Distance to Target')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Distance')
        ax2.grid(True, alpha=0.3)

    # Training time per episode
    ax3 = axes[1, 0]
    if training_metrics['training_time']:
        # Smooth the training time data
        window_size = min(50, len(training_metrics['training_time']))
        if window_size > 1:
            smoothed_times = np.convolve(training_metrics['training_time'], 
                                        np.ones(window_size)/window_size, mode='valid')
            ax3.plot(smoothed_times, 'm-', linewidth=2, alpha=0.7)
        ax3.set_title('Training Time per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True, alpha=0.3)

    # Training summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
TRAINING SUMMARY

Primary Pattern: {PRIMARY_PATTERN}
Total Episodes: {len(training_metrics['training_time'])}
Total Time: {total_training_time/60:.1f} minutes
Avg Time/Episode: {np.mean(training_metrics['training_time']):.3f}s

Best Evaluation Reward: {training_metrics['best_reward']:.2f}
Best Episode: {training_metrics['best_episode']}

Final Avg Reward: {training_metrics['rewards'][-1] if training_metrics['rewards'] else 'N/A':.2f}
Final Avg Distance: {training_metrics['distances'][-1] if training_metrics['distances'] else 'N/A':.2f}

Model Saved: models/best_dot_follow_model.pt
"""
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('training_plots/training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training progress saved to 'training_plots/training_progress.png'")

def evaluate_model(learner):
    """Comprehensive evaluation across all patterns"""
    print("\n🧪 Comprehensive Model Evaluation")
    print("="*50)
    
    eval_results = {}
    num_trials = 8  # Number of trials per pattern
    max_steps = 400  # Maximum steps per evaluation episode

    for pattern in TEST_PATTERNS:
        print(f"\nTesting on {pattern} pattern...")
        
        pattern_rewards = []
        pattern_distances = []
        pattern_episode_lengths = []
        
        for trial in range(num_trials):
            env = DotFollowEnv(pattern)
            obs = env.reset()
            
            total_reward = 0
            distances = []
            steps = 0
            
            for step in range(max_steps):
                action = learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, done, _ = env.step(action)
                
                total_reward += reward
                distance = np.linalg.norm(env.position - env.target.position)
                distances.append(distance)
                steps += 1
                
                if done:
                    break
            
            pattern_rewards.append(total_reward)
            pattern_distances.append(np.mean(distances))
            pattern_episode_lengths.append(steps)
        
        # Store results
        eval_results[pattern] = {
            'rewards': pattern_rewards,
            'distances': pattern_distances,
            'episode_lengths': pattern_episode_lengths,
            'mean_reward': np.mean(pattern_rewards),
            'std_reward': np.std(pattern_rewards),
            'mean_distance': np.mean(pattern_distances),
            'std_distance': np.std(pattern_distances),
            'mean_length': np.mean(pattern_episode_lengths),
            'std_length': np.std(pattern_episode_lengths)
        }
        
        print(f"  Reward: {eval_results[pattern]['mean_reward']:.2f} ± {eval_results[pattern]['std_reward']:.2f}")
        print(f"  Distance: {eval_results[pattern]['mean_distance']:.2f} ± {eval_results[pattern]['std_distance']:.2f}")
        print(f"  Episode Length: {eval_results[pattern]['mean_length']:.1f} ± {eval_results[pattern]['std_length']:.1f}")

    print("\n✅ Evaluation completed!")
    return eval_results

def create_evaluation_plots(eval_results):
    """Create and save evaluation comparison plots"""
    print("\n📈 Creating evaluation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    patterns = list(eval_results.keys())
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']

    # Reward comparison
    ax1 = axes[0, 0]
    mean_rewards = [eval_results[p]['mean_reward'] for p in patterns]
    std_rewards = [eval_results[p]['std_reward'] for p in patterns]

    bars1 = ax1.bar(patterns, mean_rewards, yerr=std_rewards, capsize=5, 
                    color=colors[:len(patterns)])
    ax1.set_title('Average Reward by Movement Pattern')
    ax1.set_ylabel('Reward')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Highlight the primary training pattern
    primary_idx = patterns.index(PRIMARY_PATTERN)
    bars1[primary_idx].set_color('orange')
    bars1[primary_idx].set_edgecolor('red')
    bars1[primary_idx].set_linewidth(2)

    # Distance comparison
    ax2 = axes[0, 1]
    mean_distances = [eval_results[p]['mean_distance'] for p in patterns]
    std_distances = [eval_results[p]['std_distance'] for p in patterns]

    bars2 = ax2.bar(patterns, mean_distances, yerr=std_distances, capsize=5,
                    color=colors[:len(patterns)])
    ax2.set_title('Average Distance to Target')
    ax2.set_ylabel('Distance')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Highlight the primary training pattern
    bars2[primary_idx].set_color('orange')
    bars2[primary_idx].set_edgecolor('red')
    bars2[primary_idx].set_linewidth(2)

    # Episode length comparison
    ax3 = axes[1, 0]
    mean_lengths = [eval_results[p]['mean_length'] for p in patterns]
    std_lengths = [eval_results[p]['std_length'] for p in patterns]

    bars3 = ax3.bar(patterns, mean_lengths, yerr=std_lengths, capsize=5,
                    color=colors[:len(patterns)])
    ax3.set_title('Average Episode Length')
    ax3.set_ylabel('Steps')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # Highlight the primary training pattern
    bars3[primary_idx].set_color('orange')
    bars3[primary_idx].set_edgecolor('red')
    bars3[primary_idx].set_linewidth(2)

    # Performance ranking
    ax4 = axes[1, 1]
    # Rank by reward (higher is better)
    ranked_patterns = sorted(patterns, key=lambda p: eval_results[p]['mean_reward'], reverse=True)
    ranked_rewards = [eval_results[p]['mean_reward'] for p in ranked_patterns]

    bar_colors = ['orange' if p == PRIMARY_PATTERN else 'lightblue' for p in ranked_patterns]
    bars4 = ax4.barh(range(len(ranked_patterns)), ranked_rewards, color=bar_colors)
    ax4.set_yticks(range(len(ranked_patterns)))
    ax4.set_yticklabels(ranked_patterns)
    ax4.set_title('Performance Ranking (by Reward)')
    ax4.set_xlabel('Average Reward')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (pattern, reward) in enumerate(zip(ranked_patterns, ranked_rewards)):
        ax4.text(reward + 0.5, i, f'{reward:.1f}', va='center')

    # Add legend
    ax4.text(0.02, 0.98, f'Orange = Primary training pattern ({PRIMARY_PATTERN})', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

    plt.tight_layout()
    plt.savefig('training_plots/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Evaluation results saved to 'training_plots/evaluation_results.png'")
    
    return ranked_patterns

def print_final_summary(training_metrics, eval_results, ranked_patterns, total_training_time):
    """Print comprehensive summary"""
    print("\n🎉 TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"\n📋 TRAINING SUMMARY:")
    print(f"   Primary Pattern: {PRIMARY_PATTERN}")
    print(f"   Episodes Trained: {len(training_metrics['training_time'])}")
    print(f"   Training Time: {total_training_time/60:.1f} minutes")
    print(f"   Best Training Reward: {training_metrics['best_reward']:.2f}")

    print(f"\n🏆 EVALUATION RESULTS:")
    for i, pattern in enumerate(ranked_patterns):
        reward = eval_results[pattern]['mean_reward']
        distance = eval_results[pattern]['mean_distance']
        marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        primary_marker = " (PRIMARY)" if pattern == PRIMARY_PATTERN else ""
        print(f"   {marker} {pattern:12s}: {reward:6.2f} reward, {distance:5.2f} avg distance{primary_marker}")

    print(f"\n💾 SAVED FILES:")
    print(f"   Best Model: models/best_dot_follow_model.pt")
    print(f"   Training Plot: training_plots/training_progress.png")
    print(f"   Evaluation Plot: training_plots/evaluation_results.png")

    print(f"\n🌐 WEB INTERFACE:")
    print(f"   The trained model is ready for use in the web interface!")
    print(f"   Run: python start_web_interface.py")

    # Check if model file exists and show file size
    model_path = 'models/best_dot_follow_model.pt'
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / 1024  # Size in KB
        print(f"\n✅ Model file verified: {file_size:.1f} KB")
    else:
        print(f"\n❌ Warning: Model file not found at {model_path}")

    print("\n" + "="*60)

def main():
    """Main function"""
    try:
        # Setup
        setup_directories()
        
        # Train the model
        learner, training_metrics, total_training_time = train_model()
        
        # Create training plots
        create_training_plots(training_metrics, total_training_time)
        
        # Evaluate the model
        eval_results = evaluate_model(learner)
        
        # Create evaluation plots
        ranked_patterns = create_evaluation_plots(eval_results)
        
        # Print final summary
        print_final_summary(training_metrics, eval_results, ranked_patterns, total_training_time)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Partial results may be saved in the models/ directory")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

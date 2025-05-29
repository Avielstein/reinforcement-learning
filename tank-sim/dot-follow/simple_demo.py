"""
Simple demonstration of the dot-follow environment
Shows the fish learning to follow different movement patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import sys
sys.path.append('../')

from dot_follow_environment import DotFollowEnv
from dot_follow_trainer import DotFollowLearner

def simple_training_demo(pattern='circular', episodes=50):
    """Simple training demonstration with basic visualization"""
    print(f"Training fish to follow {pattern} pattern for {episodes} episodes...")
    
    # Create learner and train
    learner = DotFollowLearner(pattern)
    
    rewards = []
    distances = []
    
    for episode in range(episodes):
        learner.train_step()
        
        if learner.ep_returns and learner.ep_target_dists:
            rewards.append(learner.ep_returns[-1])
            distances.append(learner.ep_target_dists[-1])
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={rewards[-1] if rewards else 0:.1f}, "
                  f"Avg Distance={distances[-1] if distances else 0:.1f}")
    
    # Plot training progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(rewards, 'g-', linewidth=2, alpha=0.7)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(distances, 'r-', linewidth=2, alpha=0.7)
    ax2.set_title('Average Distance to Target')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Distance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_progress_{pattern}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return learner

def demonstrate_following(learner, pattern='circular', steps=300):
    """Demonstrate the trained fish following the target"""
    print(f"Demonstrating trained fish following {pattern} pattern...")
    
    learner.load_best()
    learner.change_movement_pattern(pattern)
    
    # Reset environment and collect trajectory
    obs = learner.env.reset()
    
    fish_positions = []
    target_positions = []
    rewards = []
    
    for step in range(steps):
        # Get action from trained policy
        action = learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
        
        # Take step
        obs, reward, done, info = learner.env.step(action)
        
        # Record positions
        fish_positions.append(learner.env.position.copy())
        target_positions.append(info['target_pos'])
        rewards.append(reward)
        
        if done:
            obs = learner.env.reset()
    
    fish_positions = np.array(fish_positions)
    target_positions = np.array(target_positions)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot trajectories
    ax1.plot(target_positions[:, 0], target_positions[:, 1], 'r--', 
             linewidth=2, alpha=0.8, label='Target Path')
    ax1.plot(fish_positions[:, 0], fish_positions[:, 1], 'b-', 
             linewidth=2, alpha=0.8, label='Fish Path')
    
    # Mark start and end points
    ax1.plot(fish_positions[0, 0], fish_positions[0, 1], 'go', 
             markersize=10, label='Start')
    ax1.plot(fish_positions[-1, 0], fish_positions[-1, 1], 'ro', 
             markersize=10, label='End')
    
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    ax1.set_title(f'Fish Following {pattern.title()} Pattern')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot distance over time
    distances = [np.linalg.norm(fish_pos - target_pos) 
                for fish_pos, target_pos in zip(fish_positions, target_positions)]
    
    ax2.plot(distances, 'purple', linewidth=2, alpha=0.8)
    ax2.set_title('Distance to Target Over Time')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Distance')
    ax2.grid(True, alpha=0.3)
    
    avg_distance = np.mean(distances)
    ax2.axhline(y=avg_distance, color='red', linestyle='--', 
                label=f'Average: {avg_distance:.1f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'following_demo_{pattern}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Average distance to target: {avg_distance:.2f}")
    print(f"Average reward per step: {np.mean(rewards):.2f}")

def compare_patterns_demo():
    """Compare performance across different movement patterns"""
    patterns = ['circular', 'figure8', 'random_walk']
    
    fig, axes = plt.subplots(2, len(patterns), figsize=(15, 8))
    
    for idx, pattern in enumerate(patterns):
        print(f"\nTraining on {pattern} pattern...")
        
        # Quick training
        learner = DotFollowLearner(pattern)
        rewards = []
        
        for episode in range(30):
            learner.train_step()
            if learner.ep_returns:
                rewards.append(learner.ep_returns[-1])
        
        # Plot training curve
        ax_train = axes[0, idx]
        ax_train.plot(rewards, 'g-', linewidth=2, alpha=0.7)
        ax_train.set_title(f'{pattern.title()} Training')
        ax_train.set_xlabel('Episode')
        ax_train.set_ylabel('Reward')
        ax_train.grid(True, alpha=0.3)
        
        # Test performance
        learner.load_best()
        obs = learner.env.reset()
        
        fish_pos = []
        target_pos = []
        
        for _ in range(150):
            action = learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, done, info = learner.env.step(action)
            fish_pos.append(learner.env.position.copy())
            target_pos.append(info['target_pos'])
            if done:
                break
        
        fish_pos = np.array(fish_pos)
        target_pos = np.array(target_pos)
        
        # Plot performance
        ax_perf = axes[1, idx]
        ax_perf.plot(target_pos[:, 0], target_pos[:, 1], 'r--', 
                     linewidth=2, alpha=0.6, label='Target')
        ax_perf.plot(fish_pos[:, 0], fish_pos[:, 1], 'b-', 
                     linewidth=2, alpha=0.8, label='Fish')
        ax_perf.set_xlim(0, 100)
        ax_perf.set_ylim(0, 100)
        ax_perf.set_aspect('equal')
        ax_perf.set_title(f'{pattern.title()} Performance')
        ax_perf.grid(True, alpha=0.3)
        ax_perf.legend()
    
    plt.tight_layout()
    plt.savefig('pattern_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== Dot Follow Simple Demo ===\n")
    
    # Demo 1: Train on circular pattern
    print("1. Training on circular pattern...")
    learner = simple_training_demo('circular', episodes=40)
    
    # Demo 2: Show following behavior
    print("\n2. Demonstrating following behavior...")
    demonstrate_following(learner, 'circular', steps=200)
    
    # Demo 3: Compare different patterns
    print("\n3. Comparing different movement patterns...")
    compare_patterns_demo()
    
    print("\n=== Demo Complete! ===")
    print("Check the generated PNG files for visualizations.")

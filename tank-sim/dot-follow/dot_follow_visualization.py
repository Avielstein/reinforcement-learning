"""
Visualization functions for Dot Follow RL
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque
import threading
import queue
import torch
import sys
sys.path.append('../')

from dot_follow_trainer import DotFollowLearner
from utils.constants import *


def run_dot_follow_training(movement_pattern='circular'):
    """Run dot follow training with live visualization"""
    
    learner = DotFollowLearner(movement_pattern)
    stop_event = threading.Event()
    
    # Start training in background thread
    training_thread = threading.Thread(
        target=learner.train_forever, 
        args=(stop_event,), 
        daemon=True
    )
    training_thread.start()
    
    # Set up visualization
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[2, 1])
    
    # Tank visualization (main plot)
    ax_tank = fig.add_subplot(gs[0, 0])
    ax_tank.set_xlim(0, TANK_SIZE)
    ax_tank.set_ylim(0, TANK_SIZE)
    ax_tank.set_aspect('equal')
    ax_tank.set_title(f'Dot Follow Training - {movement_pattern.title()} Pattern')
    ax_tank.grid(True, alpha=0.3)
    
    # Fish and trail
    fish_dot, = ax_tank.plot([], [], 'bo', markersize=12, label='Fish', zorder=5)
    target_dot, = ax_tank.plot([], [], 'ro', markersize=10, label='Target', zorder=5)
    fish_trail, = ax_tank.plot([], [], 'b-', alpha=0.4, linewidth=2, label='Fish Trail')
    target_trail, = ax_tank.plot([], [], 'r--', alpha=0.6, linewidth=1, label='Target Trail')
    
    fish_trail_buf = deque(maxlen=100)
    target_trail_buf = deque(maxlen=100)
    
    # Current arrows (will be updated dynamically)
    arrows = []
    
    # Training metrics
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.set_title('Training Progress')
    ax_metrics.set_xlabel('Episodes')
    ax_metrics.set_ylabel('Reward')
    
    # Distance metrics
    ax_distance = fig.add_subplot(gs[0, 2])
    ax_distance.set_title('Target Distance')
    ax_distance.set_xlabel('Episodes')
    ax_distance.set_ylabel('Avg Distance')
    
    # Pattern controls
    ax_controls = fig.add_subplot(gs[1, :])
    ax_controls.set_title('Movement Patterns')
    ax_controls.axis('off')
    
    reward_hist = []
    dist_hist = []
    episode_hist = []
    
    # Pattern buttons
    patterns = ['circular', 'figure8', 'random_walk', 'zigzag', 'spiral']
    pattern_buttons = {}
    
    def change_pattern(pattern):
        learner.change_movement_pattern(pattern)
        ax_tank.set_title(f'Dot Follow Training - {pattern.title()} Pattern')
        # Clear trails when changing pattern
        fish_trail_buf.clear()
        target_trail_buf.clear()
    
    # Create pattern selection buttons
    button_width = 0.15
    button_height = 0.3
    for i, pattern in enumerate(patterns):
        button_ax = plt.axes([0.1 + i * 0.18, 0.05, button_width, button_height])
        button = plt.Button(button_ax, pattern.replace('_', '\n').title())
        button.on_clicked(lambda x, p=pattern: change_pattern(p))
        pattern_buttons[pattern] = button
    
    def update_frame(frame):
        # Update environment state multiple times per frame
        for _ in range(3):  # Slower visualization for better viewing
            obs = learner.env._obs()
            action = learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
            learner.env.step(action)
        
        # Update fish and target positions
        fish_pos = learner.env.position
        target_pos = learner.env.target.position
        
        fish_dot.set_data(*fish_pos)
        target_dot.set_data(*target_pos)
        
        # Update trails
        fish_trail_buf.append(fish_pos.copy())
        target_trail_buf.append(target_pos.copy())
        
        if len(fish_trail_buf) > 1:
            fish_trail.set_data(*zip(*fish_trail_buf))
        if len(target_trail_buf) > 1:
            target_trail.set_data(*zip(*target_trail_buf))
        
        # Update current arrows
        for arrow in arrows:
            arrow.remove()
        arrows.clear()
        
        for current in learner.env.currents:
            # Draw current influence area
            circle = plt.Circle(current.position, current.radius, 
                              color='cyan', alpha=0.15)
            ax_tank.add_patch(circle)
            arrows.append(circle)
            
            # Draw direction arrow
            vec = current.direction * current.strength * 3
            arrow = patches.FancyArrow(
                current.position[0], current.position[1],
                vec[0], vec[1],
                width=1.0, color='cyan', alpha=0.6
            )
            ax_tank.add_patch(arrow)
            arrows.append(arrow)
        
        # Draw connection line between fish and target
        connection_line = plt.Line2D([fish_pos[0], target_pos[0]], 
                                   [fish_pos[1], target_pos[1]], 
                                   color='gray', alpha=0.3, linestyle=':', linewidth=1)
        ax_tank.add_line(connection_line)
        arrows.append(connection_line)
        
        # Update metrics
        try:
            while True:
                mean_reward, mean_dist = learner.metric_q.get_nowait()
                reward_hist.append(mean_reward)
                dist_hist.append(mean_dist)
                episode_hist.append(len(reward_hist))
        except queue.Empty:
            pass
        
        # Plot metrics
        if reward_hist:
            ax_metrics.clear()
            ax_metrics.set_title(f'Training Progress (Episode {learner.episode_count})')
            ax_metrics.set_xlabel('Training Batches')
            ax_metrics.set_ylabel('Mean Reward')
            ax_metrics.plot(episode_hist, reward_hist, 'g-', alpha=0.7)
            ax_metrics.grid(True, alpha=0.3)
            
            ax_distance.clear()
            ax_distance.set_title('Target Distance')
            ax_distance.set_xlabel('Training Batches')
            ax_distance.set_ylabel('Avg Distance')
            ax_distance.plot(episode_hist, dist_hist, 'r-', alpha=0.7)
            ax_distance.grid(True, alpha=0.3)
        
        # Add legend to tank plot
        ax_tank.legend(loc='upper right', fontsize=8)
        
        return [fish_dot, target_dot, fish_trail, target_trail] + arrows
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update_frame, interval=100, blit=False
    )
    
    plt.tight_layout()
    
    # Keep animation alive and handle cleanup properly
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        stop_event.set()
        
        # Wait a moment for training thread to finish
        if training_thread.is_alive():
            training_thread.join(timeout=2.0)
        
        if learner.best_state:
            learner.load_best()
            model_name = f'best_dot_follow_{movement_pattern}.pt'
            learner.save_model(model_name)
            print(f'Best model saved as {model_name}! Mean return: {learner.best_return:.2f}')
        
        return learner


def test_dot_follow_model(learner, num_episodes=3, movement_patterns=None):
    """Test the trained model with different movement patterns"""
    if movement_patterns is None:
        movement_patterns = ['circular', 'figure8', 'random_walk']
    
    learner.load_best()
    
    fig, axes = plt.subplots(1, len(movement_patterns), figsize=(5*len(movement_patterns), 5))
    if len(movement_patterns) == 1:
        axes = [axes]
    
    for idx, pattern in enumerate(movement_patterns):
        ax = axes[idx]
        ax.set_xlim(0, TANK_SIZE)
        ax.set_ylim(0, TANK_SIZE)
        ax.set_aspect('equal')
        ax.set_title(f'{pattern.title()} Pattern')
        ax.grid(True, alpha=0.3)
        
        # Change to this pattern
        learner.change_movement_pattern(pattern)
        
        # Run episodes and plot trajectories
        for episode in range(num_episodes):
            obs = learner.env.reset()
            done = False
            fish_trajectory = [learner.env.position.copy()]
            target_trajectory = [learner.env.target.position.copy()]
            
            # Different color for each episode
            color = plt.cm.viridis(episode / num_episodes)
            
            while not done:
                # Get action from policy (deterministic)
                action = learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
                
                # Take step
                next_obs, _, done, info = learner.env.step(action)
                fish_trajectory.append(learner.env.position.copy())
                target_trajectory.append(info['target_pos'])
                
                obs = next_obs
            
            # Plot trajectories
            if len(fish_trajectory) > 0:
                fish_traj = np.array(fish_trajectory)
                target_traj = np.array(target_trajectory)
                
                # Plot fish trajectory
                ax.plot(fish_traj[:, 0], fish_traj[:, 1], '-', color=color, alpha=0.8, 
                       linewidth=2, label=f'Fish Ep {episode+1}')
                
                # Plot target trajectory
                ax.plot(target_traj[:, 0], target_traj[:, 1], '--', color=color, alpha=0.6, 
                       linewidth=1, label=f'Target Ep {episode+1}')
                
                # Plot start and end points
                ax.plot(fish_traj[0, 0], fish_traj[0, 1], 'o', color=color, markersize=8)
                ax.plot(fish_traj[-1, 0], fish_traj[-1, 1], 's', color=color, markersize=8)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def compare_movement_patterns():
    """Compare learning performance across different movement patterns"""
    patterns = ['circular', 'figure8', 'random_walk', 'zigzag']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, pattern in enumerate(patterns):
        print(f"Training on {pattern} pattern...")
        learner = DotFollowLearner(pattern)
        
        # Quick training run
        rewards = []
        distances = []
        
        for episode in range(50):  # Quick test
            learner.train_step()
            if learner.ep_returns:
                rewards.append(np.mean(list(learner.ep_returns)[-10:]))  # Last 10 episodes
                distances.append(np.mean(list(learner.ep_target_dists)[-10:]))
        
        ax = axes[idx]
        ax.plot(rewards, 'g-', label='Reward', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(distances, 'r-', label='Distance', alpha=0.7)
        
        ax.set_title(f'{pattern.title()} Pattern')
        ax.set_xlabel('Training Episodes')
        ax.set_ylabel('Mean Reward', color='g')
        ax2.set_ylabel('Mean Distance', color='r')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

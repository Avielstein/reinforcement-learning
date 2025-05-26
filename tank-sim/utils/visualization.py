"""
Visualization functions for Fish Tank RL
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

from .constants import *


def run_training_visualization():
    """Run training with live visualization"""
    from .trainer import A2CLearner
    
    learner = A2CLearner()
    stop_event = threading.Event()
    
    # Start training in background thread
    training_thread = threading.Thread(
        target=learner.train_forever, 
        args=(stop_event,), 
        daemon=True
    )
    training_thread.start()
    
    # Set up visualization
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 2])
    
    # Tank visualization
    ax_tank = fig.add_subplot(gs[0])
    ax_tank.set_xlim(0, TANK_SIZE)
    ax_tank.set_ylim(0, TANK_SIZE)
    ax_tank.set_aspect('equal')
    ax_tank.set_title('Fish Tank Environment')
    ax_tank.grid(True, alpha=0.3)
    
    # Add center target
    center_circle = plt.Circle(CENTER, 3, color='red', alpha=0.5)
    ax_tank.add_patch(center_circle)
    
    # Fish and trail
    fish_dot, = ax_tank.plot([], [], 'bo', markersize=10, label='Fish')
    trail, = ax_tank.plot([], [], 'b-', alpha=0.3, label='Trail')
    trail_buf = deque(maxlen=50)
    
    # Current arrows (will be updated dynamically)
    arrows = []
    
    # Metrics visualization
    ax_metrics = fig.add_subplot(gs[1])
    ax_metrics.set_title('Training Metrics')
    ax_metrics.set_xlabel('Episodes (batches)')
    
    reward_hist = []
    dist_hist = []
    
    def update_frame(frame):
        # Update environment state
        for _ in range(VISUALIZATION_INTERVAL):
            obs = learner.env._obs()
            action = learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
            learner.env.step(action)
        
        # Update fish position and trail
        pos = learner.env.position
        fish_dot.set_data(*pos)
        trail_buf.append(pos.copy())
        
        if len(trail_buf) > 1:
            trail.set_data(*zip(*trail_buf))
        
        # Update current arrows
        for arrow in arrows:
            arrow.remove()
        arrows.clear()
        
        for current in learner.env.currents:
            # Draw current influence area
            circle = plt.Circle(current.position, current.radius, 
                              color='cyan', alpha=0.2)
            ax_tank.add_patch(circle)
            arrows.append(circle)
            
            # Draw direction arrow
            vec = current.direction * current.strength * 2
            arrow = patches.FancyArrow(
                current.position[0], current.position[1],
                vec[0], vec[1],
                width=1.5, color='cyan', alpha=0.8
            )
            ax_tank.add_patch(arrow)
            arrows.append(arrow)
        
        # Update metrics
        try:
            while True:
                mean_reward, mean_dist = learner.metric_q.get_nowait()
                reward_hist.append(mean_reward)
                dist_hist.append(mean_dist)
        except queue.Empty:
            pass
        
        # Plot metrics
        ax_metrics.clear()
        ax_metrics.set_title(f'Training Metrics (Episode {learner.episode_count})')
        ax_metrics.set_xlabel('Training Batches')
        
        if reward_hist:
            ax_metrics.plot(reward_hist, 'g-', label='Mean Reward', alpha=0.7)
        if dist_hist:
            ax_metrics.plot(dist_hist, 'r-', label='Mean Distance', alpha=0.7)
        
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
        
        return [fish_dot, trail] + arrows
    
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
            learner.save_model('best_fish_policy.pt')
            print(f'Best model saved! Mean return: {learner.best_return:.2f}')
        
        return learner


def test_trained_model(learner, num_episodes=5):
    """Test the trained model and visualize trajectories"""
    learner.load_best()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, TANK_SIZE)
    ax.set_ylim(0, TANK_SIZE)
    ax.set_aspect('equal')
    ax.set_title('Trained Fish Agent Performance')
    ax.grid(True, alpha=0.3)
    
    # Add center target
    center_circle = plt.Circle(CENTER, 3, color='red', alpha=0.5)
    ax.add_patch(center_circle)
    
    # Run episodes and plot trajectories
    for episode in range(num_episodes):
        obs = learner.env.reset()
        done = False
        trajectory = [learner.env.position.copy()]
        
        # Different color for each episode trajectory
        color = plt.cm.jet(episode / num_episodes)
        
        while not done:
            # Get action from policy (deterministic)
            action = learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
            
            # Take step
            next_obs, _, done, _ = learner.env.step(action)
            trajectory.append(learner.env.position.copy())
            
            obs = next_obs
        
        # Plot trajectory
        if len(trajectory) > 0:
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, alpha=0.7, 
                   label=f'Episode {episode+1}')
            
            # Plot start and end points
            ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=color, markersize=8)
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=color, markersize=8)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Show plot
    plt.tight_layout()
    plt.show()

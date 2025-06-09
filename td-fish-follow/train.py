#!/usr/bin/env python3
"""
Main training script for TD Fish Follow.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

# Simple, self-contained implementation that works


class SimpleFishEnvironment:
    """Simple fish environment that works."""
    
    def __init__(self, pattern='circular'):
        self.tank_width = 800
        self.tank_height = 600
        self.pattern = pattern
        self.reset()
    
    def reset(self):
        """Reset environment."""
        self.fish_pos = np.array([400.0, 300.0])
        self.fish_vel = np.array([0.0, 0.0])
        self.target_pos = np.array([600.0, 200.0])
        self.step_count = 0
        self.time = 0.0
        return self._get_observation()
    
    def _get_observation(self):
        """Get observation vector."""
        # Distance to target
        dx = self.target_pos[0] - self.fish_pos[0]
        dy = self.target_pos[1] - self.fish_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Normalized direction to target
        if distance > 0:
            dir_x = dx / distance
            dir_y = dy / distance
        else:
            dir_x = dir_y = 0
        
        # Observation: [fish_x, fish_y, fish_vel_x, fish_vel_y, target_x, target_y, 
        #               distance, dir_x, dir_y, time]
        obs = np.array([
            self.fish_pos[0] / self.tank_width,
            self.fish_pos[1] / self.tank_height,
            self.fish_vel[0] / 50.0,
            self.fish_vel[1] / 50.0,
            self.target_pos[0] / self.tank_width,
            self.target_pos[1] / self.tank_height,
            distance / 500.0,
            dir_x,
            dir_y,
            np.sin(self.time * 0.1)
        ])
        return obs
    
    def step(self, action):
        """Take a step in the environment."""
        # Apply action (continuous control)
        action = np.clip(action, -1, 1)
        
        # Update fish velocity and position
        self.fish_vel = 0.8 * self.fish_vel + 0.2 * action * 20
        self.fish_pos += self.fish_vel
        
        # Keep fish in bounds
        self.fish_pos[0] = np.clip(self.fish_pos[0], 20, self.tank_width - 20)
        self.fish_pos[1] = np.clip(self.fish_pos[1], 20, self.tank_height - 20)
        
        # Update target position based on pattern
        self._update_target()
        
        # Calculate reward
        distance = np.linalg.norm(self.target_pos - self.fish_pos)
        reward = max(0, 100 - distance / 5)
        
        # Check if done
        self.step_count += 1
        self.time += 0.1
        done = self.step_count >= 200
        
        info = {'distance_to_target': distance}
        
        return self._get_observation(), reward, done, info
    
    def _update_target(self):
        """Update target position based on pattern."""
        if self.pattern == 'circular':
            center_x, center_y = 400, 300
            radius = 150
            angle = self.time * 0.02
            self.target_pos[0] = center_x + radius * np.cos(angle)
            self.target_pos[1] = center_y + radius * np.sin(angle)
        
        elif self.pattern == 'figure8':
            center_x, center_y = 400, 300
            t = self.time * 0.02
            self.target_pos[0] = center_x + 150 * np.sin(t)
            self.target_pos[1] = center_y + 75 * np.sin(2 * t)
        
        elif self.pattern == 'zigzag':
            center_x, center_y = 400, 300
            t = self.time * 0.05
            self.target_pos[0] = center_x + 200 * np.sin(t)
            self.target_pos[1] = center_y + 100 * np.sign(np.sin(t)) * (t % 1)
        
        elif self.pattern == 'spiral':
            center_x, center_y = 400, 300
            t = self.time * 0.02
            radius = 50 + t * 2
            self.target_pos[0] = center_x + radius * np.cos(t)
            self.target_pos[1] = center_y + radius * np.sin(t)
        
        elif self.pattern == 'random_walk':
            if np.random.random() < 0.02:
                self.target_pos += np.random.normal(0, 10, 2)
                self.target_pos[0] = np.clip(self.target_pos[0], 50, self.tank_width - 50)
                self.target_pos[1] = np.clip(self.target_pos[1], 50, self.tank_height - 50)


class SimplePolicyNetwork(nn.Module):
    """Simple policy network."""
    
    def __init__(self, obs_dim=10, action_dim=2, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)


class SimpleTDAgent:
    """Simple TD learning agent."""
    
    def __init__(self, obs_dim=10, action_dim=2, lr=0.001, method='td_lambda'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.method = method
        
        # Policy network
        self.policy = SimplePolicyNetwork(obs_dim, action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Value network (critic)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # TD(λ) eligibility traces
        if method == 'td_lambda':
            self.lambda_ = 0.9
            self.eligibility_traces = {}
            for param in self.critic.parameters():
                self.eligibility_traces[param] = torch.zeros_like(param)
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1
        
        # Previous action for smoothing
        self.prev_action = None
    
    def select_action(self, obs, deterministic=False):
        """Select action using policy with exploration."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if not deterministic and np.random.random() < self.epsilon:
                # Random action
                action = np.random.uniform(-1, 1, 2)
            else:
                # Policy action
                action = self.policy(obs_tensor).cpu().numpy().flatten()
        
        # Action smoothing
        if self.prev_action is not None:
            action = 0.7 * self.prev_action + 0.3 * action
        
        action = np.clip(action, -1, 1)
        self.prev_action = action.copy()
        
        return action
    
    def update(self, obs, action, reward, next_obs, done):
        """Update agent using TD learning."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)
        
        # Critic update (TD learning)
        current_value = self.critic(obs_tensor).squeeze()
        next_value = self.critic(next_obs_tensor).squeeze() if not done else torch.tensor(0.0).to(self.device)
        
        td_target = reward_tensor + 0.99 * next_value * (~done_tensor)
        td_error = td_target - current_value
        
        if self.method == 'td_lambda':
            # TD(λ) with eligibility traces
            self.critic_optimizer.zero_grad()
            current_value.backward(retain_graph=True)
            
            # Update eligibility traces
            for param in self.critic.parameters():
                if param.grad is not None:
                    self.eligibility_traces[param] = (0.99 * self.lambda_ * self.eligibility_traces[param] + 
                                                    param.grad.clone())
                    param.grad = td_error * self.eligibility_traces[param]
            
            self.critic_optimizer.step()
        else:
            # Standard TD(0)
            critic_loss = td_error.pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # Policy update (using advantage)
        policy_action = self.policy(obs_tensor)
        advantage = td_error.detach()
        
        # Simple policy loss (move toward action that gave positive advantage)
        policy_loss = -advantage * torch.sum((policy_action - action_tensor).pow(2))
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'td_error': abs(td_error.item()),
            'critic_loss': td_error.pow(2).item(),
            'policy_loss': policy_loss.item()
        }
    
    def end_episode(self):
        """End episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.prev_action = None
        
        # Reset eligibility traces for TD(λ)
        if self.method == 'td_lambda':
            for param in self.eligibility_traces:
                self.eligibility_traces[param].zero_()
    
    def save(self, filepath):
        """Save agent."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'epsilon': self.epsilon,
            'method': self.method
        }, filepath)
    
    def load(self, filepath):
        """Load agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.method = checkpoint.get('method', 'td_lambda')


def train_fish(episodes=100, pattern='circular', method='td_lambda', show_plot=True):
    """Train the fish using TD learning."""
    print(f"🐠 Training TD Fish for {episodes} episodes")
    print(f"Pattern: {pattern}, Method: {method}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("-" * 50)
    
    # Create environment and agent
    env = SimpleFishEnvironment(pattern)
    agent = SimpleTDAgent(method=method)
    
    # Training metrics
    episode_rewards = []
    episode_distances = []
    episode_td_errors = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        distances = []
        td_errors = []
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Update agent
            stats = agent.update(obs, action, reward, next_obs, done)
            
            # Track metrics
            episode_reward += reward
            distances.append(info['distance_to_target'])
            td_errors.append(stats['td_error'])
            
            obs = next_obs
        
        # End episode
        agent.end_episode()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_distances.append(np.mean(distances))
        episode_td_errors.append(np.mean(td_errors))
        
        # Print progress
        if episode % 10 == 0 or episode == episodes - 1:
            recent_rewards = episode_rewards[-5:]
            recent_distances = episode_distances[-5:]
            print(f"Episode {episode:3d}: "
                  f"Reward={episode_reward:6.1f} "
                  f"Avg_Distance={np.mean(distances):5.1f} "
                  f"TD_Error={np.mean(td_errors):.4f} "
                  f"Epsilon={agent.epsilon:.3f} "
                  f"Recent_Avg_Reward={np.mean(recent_rewards):6.1f}")
    
    # Save model
    model_path = f'trained_fish_{method}_{pattern}_{episodes}ep.pt'
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Plot results
    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        episodes_range = range(len(episode_rewards))
        
        # Rewards
        ax1.plot(episodes_range, episode_rewards, alpha=0.7, label='Episode Reward')
        ax1.plot(episodes_range, np.convolve(episode_rewards, np.ones(5)/5, mode='same'), 
                 color='red', linewidth=2, label='Moving Average')
        ax1.set_title(f'Training Rewards ({method})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distances
        ax2.plot(episodes_range, episode_distances, alpha=0.7, label='Avg Distance')
        ax2.plot(episodes_range, np.convolve(episode_distances, np.ones(5)/5, mode='same'), 
                 color='red', linewidth=2, label='Moving Average')
        ax2.set_title('Distance to Target')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Distance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # TD Errors
        ax3.plot(episodes_range, episode_td_errors, alpha=0.7, label='TD Error')
        ax3.plot(episodes_range, np.convolve(episode_td_errors, np.ones(5)/5, mode='same'), 
                 color='red', linewidth=2, label='Moving Average')
        ax3.set_title('TD Learning Errors')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('TD Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'training_results_{method}_{pattern}_{episodes}ep.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Final stats
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Method: {method}")
    print(f"Pattern: {pattern}")
    print(f"Final reward: {episode_rewards[-1]:.1f}")
    print(f"Final distance: {episode_distances[-1]:.1f}")
    print(f"Best distance: {min(episode_distances):.1f}")
    print(f"Average reward: {np.mean(episode_rewards):.1f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"{'='*50}")
    
    return agent, episode_rewards, episode_distances


def test_trained_fish(model_path, episodes=10, pattern='circular'):
    """Test a trained fish."""
    print(f"\n🧪 Testing trained fish: {model_path}")
    print(f"Episodes: {episodes}, Pattern: {pattern}")
    print("-" * 30)
    
    # Create environment and agent
    env = SimpleFishEnvironment(pattern)
    agent = SimpleTDAgent()
    agent.load(model_path)
    agent.epsilon = 0  # No exploration for testing
    
    test_rewards = []
    test_distances = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        distances = []
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            distances.append(info['distance_to_target'])
        
        avg_distance = np.mean(distances)
        test_rewards.append(episode_reward)
        test_distances.append(avg_distance)
        
        print(f"Test {episode+1:2d}: Reward={episode_reward:6.1f}, Distance={avg_distance:5.1f}")
    
    print(f"\nTest Results:")
    print(f"Average reward: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print(f"Average distance: {np.mean(test_distances):.1f} ± {np.std(test_distances):.1f}")
    print(f"Best distance: {min(test_distances):.1f}")
    
    return test_rewards, test_distances


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TD Fish training')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--pattern', type=str, default='circular', 
                       choices=['circular', 'figure8', 'random_walk', 'zigzag', 'spiral'],
                       help='Target movement pattern')
    parser.add_argument('--method', type=str, default='td_lambda',
                       choices=['td_0', 'td_lambda'],
                       help='TD learning method')
    parser.add_argument('--test', type=str, default=None, help='Test trained model')
    
    args = parser.parse_args()
    
    if args.test:
        test_trained_fish(args.test, 10, args.pattern)
    else:
        print("🐠 TD Fish Follow - Training System")
        print("=" * 50)
        
        agent, rewards, distances = train_fish(args.episodes, args.pattern, args.method)
        
        # Test the trained agent
        model_path = f'trained_fish_{args.method}_{args.pattern}_{args.episodes}ep.pt'
        test_trained_fish(model_path, 5, args.pattern)

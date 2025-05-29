"""
A2C Trainer specialized for Dot Follow Environment
"""

import numpy as np
import torch
import torch.optim as optim
from collections import deque
import queue
import sys
sys.path.append('../')

from dot_follow_environment import DotFollowEnv
from utils.models import ActorCritic
from utils.constants import *


class DotFollowLearner:
    """Advantage Actor-Critic learner for dot following task"""
    
    def __init__(self, movement_pattern='circular'):
        self.env = DotFollowEnv(movement_pattern)
        obs_dim, act_dim = self.env.obs_dim, self.env.act_dim
        
        # Slightly larger network for more complex task
        self.ac = ActorCritic(obs_dim, act_dim, hidden_size=256)
        self.pi_opt = optim.Adam(self.ac.parameters(), lr=POLICY_LR)
        self.v_opt = optim.Adam(self.ac.v.parameters(), lr=VALUE_LR)
        
        # Tracking
        self.best_return = -np.inf
        self.best_state = None
        self.ep_returns = deque(maxlen=200)
        self.ep_target_dists = deque(maxlen=200)  # Track average distance to target
        self.metric_q = queue.Queue()
        
        # Training state
        self.training = False
        self.episode_count = 0
        self.movement_pattern = movement_pattern
    
    def _gae(self, rewards, values, last_val, dones):
        """Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = last_val
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def train_step(self):
        """Single training step - train on one complete episode"""
        # Run one complete episode
        obs = self.env.reset()
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        
        ep_reward = 0
        ep_target_dist = 0
        done = False
        
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            act, logp, val = self.ac.step(obs_tensor)
            
            next_obs, reward, done, info = self.env.step(act)
            
            obs_buf.append(obs)
            act_buf.append(act)
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(val)
            done_buf.append(done)
            
            ep_reward += reward
            # Track distance to target
            target_dist = np.linalg.norm(self.env.position - self.env.target.position)
            ep_target_dist += target_dist
            
            obs = next_obs
        
        # Episode finished
        self.ep_returns.append(ep_reward)
        self.ep_target_dists.append(ep_target_dist / self.env.step_count)
        self.episode_count += 1
        
        # Only update if we have enough data
        if len(obs_buf) < 10:  # Skip very short episodes
            return
        
        # Get last value for GAE (should be 0 since episode is done)
        last_val = 0.0
        
        # Convert to numpy arrays
        rewards = np.array(rew_buf, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)
        
        # Calculate advantages and returns
        advantages, returns = self._gae(rewards, values, last_val, dones)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_tensor = torch.as_tensor(np.array(obs_buf), dtype=torch.float32)
        act_tensor = torch.as_tensor(np.array(act_buf), dtype=torch.float32)
        logp_tensor = torch.as_tensor(np.array(logp_buf), dtype=torch.float32)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32)
        
        # Update policy and value function
        for _ in range(TRAIN_ITERS):
            # Policy update
            mu = self.ac.pi(obs_tensor)
            std = self.ac.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            new_logp = dist.log_prob(act_tensor).sum(-1)
            ratio = torch.exp(new_logp - logp_tensor)
            
            # Clipped surrogate objective (PPO-style)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv_tensor
            pi_loss = -torch.min(surr1, surr2).mean()
            
            self.pi_opt.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.pi_opt.step()
            
            # Value update
            v_pred = self.ac.v(obs_tensor).squeeze(-1)
            v_loss = ((v_pred - ret_tensor) ** 2).mean()
            
            self.v_opt.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.v.parameters(), 0.5)
            self.v_opt.step()
        
        # Update best model
        mean_return = np.mean(self.ep_returns) if self.ep_returns else -np.inf
        if mean_return > self.best_return:
            self.best_return = mean_return
            self.best_state = self.ac.state_dict()
        
        # Send metrics for visualization
        if not self.metric_q.full():
            mean_dist = np.mean(self.ep_target_dists) if self.ep_target_dists else np.nan
            self.metric_q.put((mean_return, mean_dist))
    
    def train_forever(self, stop_event):
        """Training loop"""
        self.training = True
        while not stop_event.is_set():
            self.train_step()
        self.training = False
    
    def load_best(self):
        """Load best model"""
        if self.best_state:
            self.ac.load_state_dict(self.best_state)
    
    def save_model(self, path):
        """Save current model"""
        torch.save(self.ac.state_dict(), path)
    
    def load_model(self, path):
        """Load model from file"""
        self.ac.load_state_dict(torch.load(path))
    
    def change_movement_pattern(self, pattern):
        """Change the target movement pattern"""
        self.movement_pattern = pattern
        self.env.set_movement_pattern(pattern)
        print(f"Changed movement pattern to: {pattern}")
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        if not self.ep_returns or not self.ep_target_dists:
            return None
        
        return {
            'mean_reward': np.mean(self.ep_returns),
            'mean_target_distance': np.mean(self.ep_target_dists),
            'episodes_trained': self.episode_count,
            'movement_pattern': self.movement_pattern
        }

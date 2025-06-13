"""
Base DQN Agent - Foundation for all DQN variants
Provides common functionality and interface for DQN algorithms
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class ReplayBuffer:
    """Efficient numpy-based replay buffer for DQN variants"""
    
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.int64)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )

    def __len__(self) -> int:
        return self.size


class DQNNetwork(nn.Module):
    """Standard DQN network architecture"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, act_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BaseDQNAgent:
    """
    Base DQN Agent - Abstract class for all DQN variants
    
    Provides common functionality:
    - Experience replay
    - Target network updates
    - Epsilon-greedy exploration
    - Training loop management
    - Performance tracking
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: str = "auto"
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network
        self.update_target_network()
        
        # Replay buffer
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        
        # Training tracking
        self.step_count = 0
        self.episode_count = 0
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'q_values': [],
            'epsilons': []
        }
    
    def _build_network(self) -> nn.Module:
        """Build the Q-network architecture - override in subclasses"""
        return DQNNetwork(self.obs_dim, self.act_dim)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.act_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.store(state, action, reward, next_state, done)
    
    def update_target_network(self):
        """Hard update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def compute_loss(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute DQN loss - override in subclasses for variants"""
        states = torch.FloatTensor(batch['obs']).to(self.device)
        actions = torch.LongTensor(batch['acts']).to(self.device)
        rewards = torch.FloatTensor(batch['rews']).to(self.device)
        next_states = torch.FloatTensor(batch['next_obs']).to(self.device)
        dones = torch.FloatTensor(batch['done']).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        return loss
    
    def update(self) -> Optional[float]:
        """Update the Q-network"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.memory.sample_batch()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Track metrics
        self.training_history['losses'].append(loss.item())
        self.training_history['epsilons'].append(self.epsilon)
        
        return loss.item()
    
    def train_episode(self, env) -> Tuple[float, int]:
        """Train for one episode"""
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle new gym API
        
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
                truncated = False
            else:
                next_state, reward, done, truncated, info = result
            
            # Store transition
            self.store_transition(state, action, reward, next_state, done or truncated)
            
            # Update
            loss = self.update()
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            self.step_count += 1
            
            if done or truncated:
                break
            
            state = next_state
        
        # Record episode metrics
        self.episode_count += 1
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['episode_lengths'].append(episode_length)
        
        return episode_reward, episode_length
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent"""
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.select_action(state, training=False)
                result = env.step(action)
                
                if len(result) == 4:
                    next_state, reward, done, info = result
                    truncated = False
                else:
                    next_state, reward, done, truncated, info = result
                
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
                
                state = next_state
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths)
        }
    
    def plot_training_progress(self, window_size: int = 100):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        rewards = self.training_history['episode_rewards']
        if len(rewards) > window_size:
            smoothed_rewards = [np.mean(rewards[i:i+window_size]) 
                              for i in range(len(rewards) - window_size + 1)]
            axes[0, 0].plot(smoothed_rewards)
            axes[0, 0].set_title(f'Episode Rewards (smoothed over {window_size} episodes)')
        else:
            axes[0, 0].plot(rewards)
            axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Losses
        if self.training_history['losses']:
            axes[0, 1].plot(self.training_history['losses'])
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Loss')
        
        # Epsilon decay
        if self.training_history['epsilons']:
            axes[1, 0].plot(self.training_history['epsilons'])
            axes[1, 0].set_title('Epsilon Decay')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Epsilon')
        
        # Episode lengths
        lengths = self.training_history['episode_lengths']
        if len(lengths) > window_size:
            smoothed_lengths = [np.mean(lengths[i:i+window_size]) 
                               for i in range(len(lengths) - window_size + 1)]
            axes[1, 1].plot(smoothed_lengths)
            axes[1, 1].set_title(f'Episode Lengths (smoothed over {window_size} episodes)')
        else:
            axes[1, 1].plot(lengths)
            axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Length')
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str):
        """Save the agent"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'hyperparameters': {
                'obs_dim': self.obs_dim,
                'act_dim': self.act_dim,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }, filepath)
    
    def load(self, filepath: str):
        """Load the agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.epsilon = hyperparams['epsilon']

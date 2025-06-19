"""RAINBOW DQN algorithm implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from typing import Tuple, Dict, Any

from agent.networks import RainbowNetwork
from agent.replay_buffer import PrioritizedReplayBuffer

class RainbowDQN:
    """RAINBOW DQN agent implementation combining multiple DQN improvements."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        batch_size: int = 32,
        buffer_size: int = 10000,
        hidden_dims: list = [512, 512],
        # RAINBOW specific parameters
        n_step: int = 3,  # Multi-step learning
        v_min: float = -10.0,  # Distributional RL
        v_max: float = 10.0,
        n_atoms: int = 51,
        noisy_std: float = 0.5,  # Noisy networks
        device: str = None
    ):
        """
        Initialize RAINBOW DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate (for epsilon-greedy fallback)
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            target_update_freq: Frequency of target network updates
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            hidden_dims: Hidden layer dimensions
            n_step: Number of steps for multi-step learning
            v_min: Minimum value for distributional RL
            v_max: Maximum value for distributional RL
            n_atoms: Number of atoms for distributional RL
            noisy_std: Standard deviation for noisy networks
            device: Device to run on (cuda/cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # RAINBOW specific parameters
        self.n_step = n_step
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self.noisy_std = noisy_std
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Networks (RAINBOW combines Dueling + Noisy + Distributional)
        self.q_network = RainbowNetwork(
            state_dim, action_dim, hidden_dims, n_atoms, noisy_std
        ).to(self.device)
        self.target_network = RainbowNetwork(
            state_dim, action_dim, hidden_dims, n_atoms, noisy_std
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Multi-step buffer for n-step learning
        self.n_step_buffer = []
        
        # Training tracking
        self.steps = 0
        self.episodes = 0
        self.losses = []
        self.episode_rewards = []
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Get action using noisy networks (no epsilon-greedy needed).
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                # Use noisy networks for exploration
                dist = self.q_network(state_tensor)
                q_values = (dist * self.support).sum(2)
            else:
                # Disable noise for evaluation
                self.q_network.eval()
                dist = self.q_network(state_tensor)
                q_values = (dist * self.support).sum(2)
                self.q_network.train()
            
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience with n-step learning."""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If buffer is full, compute n-step return and store
        if len(self.n_step_buffer) >= self.n_step:
            # Compute n-step return
            n_step_reward = 0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:  # Episode ended
                    break
            
            # Get initial state and action
            init_state, init_action = self.n_step_buffer[0][:2]
            
            # Get final state and done flag
            final_state, final_done = self.n_step_buffer[-1][3:]
            
            # Store n-step experience
            self.replay_buffer.push(
                init_state, init_action, n_step_reward, final_state, final_done
            )
            
            # Remove oldest experience
            self.n_step_buffer.pop(0)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step with RAINBOW improvements.
        
        Returns:
            Training metrics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch with prioritized replay
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Current distribution
        current_dist = self.q_network(states)
        current_dist = current_dist[range(self.batch_size), actions]
        
        # Target distribution (Double DQN + Distributional)
        with torch.no_grad():
            # Use online network to select actions (Double DQN)
            next_dist = self.q_network(next_states)
            next_q_values = (next_dist * self.support).sum(2)
            next_actions = next_q_values.argmax(1)
            
            # Use target network to evaluate (Double DQN)
            target_dist = self.target_network(next_states)
            target_dist = target_dist[range(self.batch_size), next_actions]
            
            # Compute target support
            target_support = rewards.unsqueeze(1) + \
                           (self.gamma ** self.n_step) * self.support.unsqueeze(0) * \
                           (~dones).unsqueeze(1)
            target_support = target_support.clamp(self.v_min, self.v_max)
            
            # Distribute probability
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Fix disappearing probability mass
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) * (l == u)] += 1
            
            # Distribute probability mass
            target_prob = torch.zeros_like(target_dist)
            target_prob.scatter_add_(1, l, target_dist * (u.float() - b))
            target_prob.scatter_add_(1, u, target_dist * (b - l.float()))
        
        # Compute loss (cross-entropy)
        loss = -(target_prob * current_dist.log()).sum(1)
        
        # Apply importance sampling weights
        loss = (weights * loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        td_errors = (target_prob * (self.support.unsqueeze(0) - 
                    (current_dist * self.support).sum(1, keepdim=True))).sum(1)
        priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Reset noisy layers
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
        # Store loss
        self.losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'q_value_mean': (current_dist * self.support).sum(1).mean().item(),
            'target_q_mean': (target_prob * self.support).sum(1).mean().item()
        }
    
    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'n_step': self.n_step,
                'v_min': self.v_min,
                'v_max': self.v_max,
                'n_atoms': self.n_atoms,
                'noisy_std': self.noisy_std
            }
        }, filepath)
    
    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.losses = checkpoint['losses']
        self.episode_rewards = checkpoint['episode_rewards']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'recent_losses': self.losses[-20:],
            'recent_rewards': self.episode_rewards[-20:]
        }

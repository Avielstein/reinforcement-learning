"""Double DQN algorithm implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Tuple, Dict, Any

from agent.networks import QNetwork, DuelingQNetwork
from agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DoubleDQN:
    """Double DQN agent implementation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        batch_size: int = 32,
        buffer_size: int = 10000,
        hidden_dims: list = [128, 128],
        use_dueling: bool = False,
        use_prioritized_replay: bool = False,
        device: str = None
    ):
        """
        Initialize Double DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            target_update_freq: Frequency of target network updates
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            hidden_dims: Hidden layer dimensions
            use_dueling: Whether to use dueling architecture
            use_prioritized_replay: Whether to use prioritized replay
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
        self.use_prioritized_replay = use_prioritized_replay
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        NetworkClass = DuelingQNetwork if use_dueling else QNetwork
        self.q_network = NetworkClass(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = NetworkClass(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.steps = 0
        self.episodes = 0
        self.losses = []
        self.episode_rewards = []
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects epsilon)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Training metrics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, weights, indices = \
                self.replay_buffer.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, priorities + 1e-6)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Store loss
        self.losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'q_value_mean': current_q_values.mean().item(),
            'target_q_mean': target_q_values.mean().item()
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
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'target_update_freq': self.target_update_freq,
                'batch_size': self.batch_size,
                'use_prioritized_replay': self.use_prioritized_replay
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

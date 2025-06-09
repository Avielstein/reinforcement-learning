"""
Experience replay buffer for TD learning.
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from collections import namedtuple

# Experience tuple for storing transitions
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 'td_error'
])


class ReplayBuffer:
    """Circular buffer for storing and sampling experiences."""
    
    def __init__(self, capacity: int, prioritized: bool = False, alpha: float = 0.6):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            prioritized: Whether to use prioritized experience replay
            alpha: Prioritization strength (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        
        # Storage
        self.buffer = []
        self.position = 0
        
        # For prioritized replay
        if self.prioritized:
            self.priorities = np.zeros(capacity, dtype=np.float32)
            self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool, td_error: float = 1.0):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done, td_error)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set priority for new experience
        if self.prioritized:
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[self.position] = priority
            self.max_priority = max(self.max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling correction strength
            
        Returns:
            batch: List of sampled experiences
            indices: Indices of sampled experiences (for priority updates)
            weights: Importance sampling weights
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences in buffer: {len(self.buffer)} < {batch_size}")
        
        if self.prioritized:
            return self._sample_prioritized(batch_size, beta)
        else:
            return self._sample_uniform(batch_size)
    
    def _sample_uniform(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample uniformly from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)  # Uniform weights
        return batch, indices, weights
    
    def _sample_prioritized(self, batch_size: int, beta: float) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample using prioritized experience replay."""
        # Get valid priorities
        valid_size = len(self.buffer)
        priorities = self.priorities[:valid_size]
        
        # Sample indices based on priorities
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(valid_size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (valid_size * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize by max weight
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled experiences."""
        if not self.prioritized:
            return
        
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.position = 0
        if self.prioritized:
            self.priorities.fill(0)
            self.max_priority = 1.0
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        stats = {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity
        }
        
        if self.prioritized and len(self.buffer) > 0:
            valid_priorities = self.priorities[:len(self.buffer)]
            stats.update({
                'max_priority': self.max_priority,
                'mean_priority': np.mean(valid_priorities),
                'std_priority': np.std(valid_priorities)
            })
        
        return stats


class SumTree:
    """Sum tree data structure for efficient prioritized sampling."""
    
    def __init__(self, capacity: int):
        """Initialize sum tree."""
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index based on priority sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """Add new data with priority."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority of existing data."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """Get data based on priority sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Advanced prioritized replay buffer using sum tree."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization strength
            beta: Importance sampling correction
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.max_beta = 1.0
        
        self.epsilon = 1e-6  # Small constant to avoid zero priorities
        self.abs_err_upper = 1.0  # Upper bound for TD error
    
    def add(self, experience: Experience):
        """Add experience with maximum priority."""
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        
        self.tree.add(max_priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        batch = []
        indices = np.empty(batch_size, dtype=np.int32)
        weights = np.empty(batch_size, dtype=np.float32)
        
        # Calculate priority segment size
        priority_segment = self.tree.total() / batch_size
        
        # Update beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        # Calculate minimum probability for importance sampling
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        max_weight = (min_prob * self.tree.n_entries) ** (-self.beta)
        
        for i in range(batch_size):
            # Sample from priority segment
            a = priority_segment * i
            b = priority_segment * (i + 1)
            s = np.random.uniform(a, b)
            
            # Get experience
            idx, priority, experience = self.tree.get(s)
            
            # Calculate importance sampling weight
            prob = priority / self.tree.total()
            weight = (prob * self.tree.n_entries) ** (-self.beta)
            weight = weight / max_weight
            
            batch.append(experience)
            indices[i] = idx
            weights[i] = weight
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            priority = min(priority, self.abs_err_upper)
            self.tree.update(idx, priority)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.tree.n_entries

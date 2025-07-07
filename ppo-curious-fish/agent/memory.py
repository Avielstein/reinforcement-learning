import numpy as np
import torch

class PPOMemory:
    """Memory buffer for PPO algorithm with curiosity support."""
    
    def __init__(self, buffer_size=2048, state_dim=152, action_dim=4, device='cpu'):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.intrinsic_rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        
        # Buffer management
        self.ptr = 0
        self.size = 0
        
    def store(self, state, action, reward, intrinsic_reward, value, log_prob, done, next_state):
        """Store a single transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.intrinsic_rewards[self.ptr] = intrinsic_reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.next_states[self.ptr] = next_state
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get_batch(self, batch_size=None):
        """Get a batch of experiences."""
        if batch_size is None:
            batch_size = self.size
        
        # Get indices for the batch
        if batch_size >= self.size:
            indices = np.arange(self.size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'intrinsic_rewards': torch.FloatTensor(self.intrinsic_rewards[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device)
        }
    
    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95, curiosity_weight=0.1):
        """
        Compute Generalized Advantage Estimation (GAE) with curiosity rewards.
        
        Args:
            next_value: Value of the next state after the last stored state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            curiosity_weight: Weight for intrinsic rewards
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        # Combine extrinsic and intrinsic rewards
        total_rewards = self.rewards[:self.size] + curiosity_weight * self.intrinsic_rewards[:self.size]
        
        advantages = np.zeros(self.size, dtype=np.float32)
        returns = np.zeros(self.size, dtype=np.float32)
        
        # Bootstrap value for the last state
        if self.size > 0:
            last_gae_lam = 0
            
            # Work backwards through the buffer
            for step in reversed(range(self.size)):
                if step == self.size - 1:
                    next_non_terminal = 1.0 - self.dones[step]
                    next_value_step = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[step]
                    next_value_step = self.values[step + 1]
                
                # TD error
                delta = total_rewards[step] + gamma * next_value_step * next_non_terminal - self.values[step]
                
                # GAE
                advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
            # Returns are advantages + values
            returns = advantages + self.values[:self.size]
        
        return advantages, returns
    
    def get_all_data(self):
        """Get all stored data."""
        return {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'intrinsic_rewards': self.intrinsic_rewards[:self.size],
            'values': self.values[:self.size],
            'log_probs': self.log_probs[:self.size],
            'dones': self.dones[:self.size],
            'next_states': self.next_states[:self.size]
        }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
    
    def is_full(self):
        """Check if buffer is full."""
        return self.size >= self.buffer_size
    
    def __len__(self):
        return self.size
    
    def get_statistics(self):
        """Get buffer statistics."""
        if self.size == 0:
            return {
                'size': 0,
                'avg_reward': 0.0,
                'avg_intrinsic_reward': 0.0,
                'avg_value': 0.0
            }
        
        return {
            'size': self.size,
            'avg_reward': np.mean(self.rewards[:self.size]),
            'avg_intrinsic_reward': np.mean(self.intrinsic_rewards[:self.size]),
            'avg_value': np.mean(self.values[:self.size]),
            'max_reward': np.max(self.rewards[:self.size]),
            'min_reward': np.min(self.rewards[:self.size]),
            'reward_std': np.std(self.rewards[:self.size])
        }

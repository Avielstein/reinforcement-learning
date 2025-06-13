"""
Double DQN Implementation
Paper: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)

Key Innovation: Separate action selection and evaluation to reduce overestimation bias
- Use main network to select actions
- Use target network to evaluate selected actions
- Reduces overoptimistic value estimates that plague vanilla DQN
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from .base_dqn import BaseDQNAgent


class DoubleDQNAgent(BaseDQNAgent):
    """
    Double DQN Agent
    
    Addresses the overestimation bias in vanilla DQN by decoupling action selection
    from action evaluation. This leads to more stable and accurate Q-value estimates.
    
    Algorithm:
    1. Use main Q-network to select best action: a* = argmax_a Q(s', a; θ)
    2. Use target network to evaluate that action: Q(s', a*; θ-)
    3. This prevents the same network from both selecting and evaluating actions
    """
    
    def __init__(self, obs_dim: int, act_dim: int, **kwargs):
        super().__init__(obs_dim, act_dim, **kwargs)
        self.algorithm_name = "Double DQN"
    
    def compute_loss(self, batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Compute Double DQN loss
        
        Key difference from vanilla DQN:
        - Action selection: argmax_a Q(s', a; θ) using main network
        - Action evaluation: Q(s', a*; θ-) using target network
        """
        states = torch.FloatTensor(batch['obs']).to(self.device)
        actions = torch.LongTensor(batch['acts']).to(self.device)
        rewards = torch.FloatTensor(batch['rews']).to(self.device)
        next_states = torch.FloatTensor(batch['next_obs']).to(self.device)
        dones = torch.FloatTensor(batch['done']).to(self.device)
        
        # Current Q values: Q(s, a; θ)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            # Double DQN: Use main network for action selection
            next_q_values_main = self.q_network(next_states)
            next_actions = next_q_values_main.argmax(1)
            
            # Use target network for action evaluation
            next_q_values_target = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target: r + γ * Q(s', argmax_a Q(s', a; θ); θ-)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        return loss
    
    def get_algorithm_info(self) -> Dict[str, str]:
        """Return information about the algorithm"""
        return {
            'name': 'Double DQN',
            'paper': 'van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)',
            'key_innovation': 'Separate action selection and evaluation to reduce overestimation bias',
            'benefits': [
                'Reduces overoptimistic Q-value estimates',
                'More stable learning, especially early in training',
                'Better performance in environments with stochastic rewards',
                'Minimal computational overhead compared to vanilla DQN'
            ],
            'when_to_use': [
                'When vanilla DQN shows signs of overestimation',
                'In environments with noisy or stochastic rewards',
                'As a general improvement over vanilla DQN',
                'When training stability is important'
            ]
        }


# Convenience function for easy instantiation
def create_double_dqn_agent(env, **kwargs):
    """Create a Double DQN agent for the given environment"""
    if hasattr(env.observation_space, 'shape'):
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = env.observation_space.n
    
    act_dim = env.action_space.n
    
    return DoubleDQNAgent(obs_dim, act_dim, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    import gymnasium as gym
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Create Double DQN agent
    agent = create_double_dqn_agent(env, lr=1e-3, gamma=0.99)
    
    print("Double DQN Agent created successfully!")
    print(f"Observation dimension: {agent.obs_dim}")
    print(f"Action dimension: {agent.act_dim}")
    print(f"Device: {agent.device}")
    
    # Print algorithm info
    info = agent.get_algorithm_info()
    print(f"\nAlgorithm: {info['name']}")
    print(f"Paper: {info['paper']}")
    print(f"Key Innovation: {info['key_innovation']}")
    
    # Test a single step
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    action = agent.select_action(state)
    print(f"\nSelected action: {action}")
    
    # Test training step
    next_state, reward, done, truncated, info = env.step(action)
    agent.store_transition(state, action, reward, next_state, done or truncated)
    
    print("Single training step completed successfully!")
    
    env.close()

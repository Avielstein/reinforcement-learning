"""
Neural Network Models for Fish Tank RL
"""

import torch
import torch.nn as nn


def mlp(sizes, act=nn.Tanh, last=nn.Identity):
    """Create a multi-layer perceptron"""
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i+1]), 
                  (act if i < len(sizes) - 2 else last)()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor-Critic network for A2C algorithm"""
    
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        
        # Policy network (actor)
        self.pi = mlp([obs_dim, hidden_size, hidden_size, act_dim], last=nn.Tanh)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Value network (critic)
        self.v = mlp([obs_dim, hidden_size, hidden_size, 1])
    
    def step(self, obs):
        """Take a step (sample action and get value)"""
        with torch.no_grad():
            mu = self.pi(obs)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)
            val = self.v(obs).squeeze(-1)
        return act.detach().cpu().numpy(), logp.detach().cpu().numpy(), val.detach().cpu().numpy()
    
    def act(self, obs):
        """Get action only (for evaluation)"""
        with torch.no_grad():
            mu = self.pi(obs)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            act = dist.sample()
        return act.detach().cpu().numpy()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticNetwork(nn.Module):
    """
    A3C Actor-Critic network with shared feature extraction.
    
    The network has:
    - Shared feature layers for both actor and critic
    - Actor head outputting action probabilities
    - Critic head outputting state value
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value network)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            action_logits: Raw action logits from actor
            value: State value from critic
        """
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Actor output (action logits)
        action_logits = self.actor_head(features)
        
        # Critic output (state value)
        value = self.critic_head(features)
        
        return action_logits, value
    
    def get_action_and_value(self, state, action=None):
        """
        Get action probabilities, sampled action, and value.
        
        Args:
            state: Input state tensor
            action: Optional specific action to evaluate
            
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of the action
            entropy: Policy entropy
            value: State value
        """
        action_logits, value = self.forward(state)
        
        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        # Sample action if not provided
        if action is None:
            action = action_dist.sample()
        
        # Calculate log probability and entropy
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, state):
        """Get only the value estimate for a state."""
        _, value = self.forward(state)
        return value.squeeze(-1)


class TrustRegionActorCritic(ActorCriticNetwork):
    """
    Extended Actor-Critic network with trust region capabilities.
    
    Adds functionality for:
    - KL divergence calculation between old and new policies
    - Trust region constraint checking
    - Adaptive learning rate based on KL divergence
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_kl=0.01):
        super(TrustRegionActorCritic, self).__init__(state_dim, action_dim, hidden_dim)
        
        self.max_kl = max_kl
        self.adaptive_lr = True
        
    def compute_kl_divergence(self, old_logits, new_logits):
        """
        Compute KL divergence between old and new policy distributions.
        
        Args:
            old_logits: Action logits from old policy
            new_logits: Action logits from new policy
            
        Returns:
            kl_div: KL divergence between distributions
        """
        old_dist = torch.distributions.Categorical(logits=old_logits)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        
        kl_div = torch.distributions.kl_divergence(old_dist, new_dist)
        return kl_div.mean()
    
    def trust_region_update(self, states, actions, old_logits, advantages, returns, optimizer):
        """
        Perform trust region constrained policy update.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            old_logits: Action logits from old policy
            advantages: Advantage estimates
            returns: Discounted returns
            optimizer: PyTorch optimizer
            
        Returns:
            policy_loss: Policy loss value
            value_loss: Value loss value
            kl_div: KL divergence
            update_applied: Whether update was applied
        """
        # Forward pass with current policy
        new_logits, values = self.forward(states)
        
        # Calculate KL divergence
        kl_div = self.compute_kl_divergence(old_logits, new_logits)
        
        # Calculate losses
        policy_loss = self._calculate_policy_loss(new_logits, actions, advantages)
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # Check trust region constraint
        if kl_div.item() > self.max_kl:
            # KL divergence too high, skip update
            return policy_loss.item(), value_loss.item(), kl_div.item(), False
        
        # Apply update
        total_loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        return policy_loss.item(), value_loss.item(), kl_div.item(), True
    
    def _calculate_policy_loss(self, logits, actions, advantages):
        """Calculate policy loss using advantage estimates."""
        action_dist = torch.distributions.Categorical(logits=logits)
        log_probs = action_dist.log_prob(actions)
        
        # Policy gradient loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Add entropy bonus for exploration
        entropy = action_dist.entropy().mean()
        policy_loss -= 0.01 * entropy
        
        return policy_loss


class SharedGlobalNetwork(TrustRegionActorCritic):
    """
    Global network shared across all A3C workers.
    
    This network is updated asynchronously by multiple worker threads
    and serves as the central repository of learned knowledge.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_kl=0.01):
        super(SharedGlobalNetwork, self).__init__(state_dim, action_dim, hidden_dim, max_kl)
        
        # Make parameters shared across processes
        self.share_memory()
        
        # Track update statistics
        self.update_count = 0
        self.total_kl_divergence = 0.0
        self.successful_updates = 0
        
    def async_update(self, local_gradients, learning_rate=3e-4):
        """
        Apply gradients from a local worker to the global network.
        
        Args:
            local_gradients: List of gradients from local worker
            learning_rate: Learning rate for the update
        """
        # Apply gradients to global parameters
        for global_param, local_grad in zip(self.parameters(), local_gradients):
            if local_grad is not None:
                global_param.grad = local_grad
        
        # Update parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer.step()
        optimizer.zero_grad()
        
        self.update_count += 1
    
    def get_shared_parameters(self):
        """Get current parameters for copying to local workers."""
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def load_shared_parameters(self, shared_params):
        """Load parameters from shared dictionary."""
        for name, param in self.named_parameters():
            param.data.copy_(shared_params[name])


def create_networks(state_dim, action_dim, hidden_dim=256, max_kl=0.01):
    """
    Factory function to create A3C networks.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        max_kl: Maximum KL divergence for trust region
        
    Returns:
        global_net: Shared global network
        local_net: Local worker network
    """
    global_net = SharedGlobalNetwork(state_dim, action_dim, hidden_dim, max_kl)
    local_net = TrustRegionActorCritic(state_dim, action_dim, hidden_dim, max_kl)
    
    return global_net, local_net


if __name__ == "__main__":
    # Test the networks
    state_dim = 152  # Same as PPO curious fish (30 rays * 5 + 2 proprioception)
    action_dim = 4   # Up, Down, Left, Right
    
    # Create networks
    global_net, local_net = create_networks(state_dim, action_dim)
    
    # Test forward pass
    test_state = torch.randn(1, state_dim)
    action, log_prob, entropy, value = local_net.get_action_and_value(test_state)
    
    print(f"Network test successful!")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Action: {action.item()}, Value: {value.item():.3f}")
    print(f"Log prob: {log_prob.item():.3f}, Entropy: {entropy.item():.3f}")
    
    # Test trust region functionality
    old_logits, _ = local_net.forward(test_state)
    new_logits, _ = local_net.forward(test_state)
    kl_div = local_net.compute_kl_divergence(old_logits, new_logits)
    print(f"KL divergence: {kl_div.item():.6f}")

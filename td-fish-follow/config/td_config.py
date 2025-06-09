"""
TD Learning specific configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TDConfig:
    """Configuration for TD learning algorithms."""
    
    # TD Method selection
    method: str = 'td_lambda'  # 'td_0', 'td_lambda', 'n_step_td'
    
    # TD(λ) parameters
    lambda_param: float = 0.9  # Eligibility trace decay
    eligibility_trace_decay: float = 0.95
    replace_traces: bool = True  # Replace vs accumulate traces
    
    # N-step TD parameters
    n_steps: int = 5  # Number of steps for n-step TD
    
    # Learning rates
    value_lr: float = 3e-4  # Value function learning rate
    policy_lr: float = 1e-4  # Policy learning rate
    lr_decay: float = 0.999  # Learning rate decay per episode
    min_lr: float = 1e-6  # Minimum learning rate
    
    # Experience replay
    use_replay: bool = True
    replay_buffer_size: int = 100000
    batch_size: int = 64
    replay_start_size: int = 1000  # Min experiences before replay
    prioritized_replay: bool = True
    priority_alpha: float = 0.6  # Prioritization strength
    priority_beta: float = 0.4  # Importance sampling correction
    priority_beta_increment: float = 0.001
    
    # Target network (for stability)
    use_target_network: bool = True
    target_update_frequency: int = 100  # Steps between target updates
    soft_update_tau: float = 0.005  # Soft update rate
    
    # Discount factor
    gamma: float = 0.99
    
    # Exploration
    exploration_method: str = 'epsilon_greedy'  # 'epsilon_greedy', 'gaussian_noise'
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    gaussian_noise_std: float = 0.1
    noise_decay: float = 0.999
    
    # Value function approximation
    value_network_hidden: list = None
    use_dueling: bool = False  # Dueling network architecture
    
    # Policy network
    policy_network_hidden: list = None
    policy_activation: str = 'tanh'  # Final activation for continuous actions
    
    # Training stability
    gradient_clip_norm: float = 1.0
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01  # For policy entropy regularization
    
    # TD error clipping
    clip_td_error: bool = True
    td_error_clip: float = 1.0
    
    # Debugging and monitoring
    log_td_errors: bool = True
    log_eligibility_traces: bool = False
    save_value_function: bool = True
    
    def __post_init__(self):
        """Set default network architectures if not provided."""
        if self.value_network_hidden is None:
            self.value_network_hidden = [128, 128, 64]
        
        if self.policy_network_hidden is None:
            self.policy_network_hidden = [128, 128]
    
    def get_method_params(self) -> Dict[str, Any]:
        """Get parameters specific to the selected TD method."""
        if self.method == 'td_lambda':
            return {
                'lambda': self.lambda_param,
                'eligibility_decay': self.eligibility_trace_decay,
                'replace_traces': self.replace_traces
            }
        elif self.method == 'n_step_td':
            return {
                'n_steps': self.n_steps
            }
        elif self.method == 'td_0':
            return {}
        else:
            raise ValueError(f"Unknown TD method: {self.method}")
    
    def get_exploration_params(self) -> Dict[str, Any]:
        """Get parameters for the exploration strategy."""
        if self.exploration_method == 'epsilon_greedy':
            return {
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay
            }
        elif self.exploration_method == 'gaussian_noise':
            return {
                'noise_std': self.gaussian_noise_std,
                'noise_decay': self.noise_decay
            }
        else:
            raise ValueError(f"Unknown exploration method: {self.exploration_method}")
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        assert 0 <= self.lambda_param <= 1, "Lambda must be in [0, 1]"
        assert 0 <= self.gamma <= 1, "Gamma must be in [0, 1]"
        assert self.n_steps > 0, "N-steps must be positive"
        assert self.value_lr > 0, "Value learning rate must be positive"
        assert self.policy_lr > 0, "Policy learning rate must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        
        if self.prioritized_replay:
            assert 0 <= self.priority_alpha <= 1, "Priority alpha must be in [0, 1]"
            assert 0 <= self.priority_beta <= 1, "Priority beta must be in [0, 1]"
        
        return True

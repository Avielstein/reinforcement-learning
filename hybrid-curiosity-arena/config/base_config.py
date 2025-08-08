"""
Base configuration for the Hybrid Multi-Agent Curiosity Arena
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class AgentType(Enum):
    """Types of agents in the arena"""
    CURIOUS = "curious"          # PPO + ICM
    COMPETITIVE = "competitive"  # A3C
    HYBRID = "hybrid"           # Switches between strategies
    ADAPTIVE = "adaptive"       # Learns which strategy to use


class RewardType(Enum):
    """Types of reward systems"""
    CURIOSITY = "curiosity"     # Intrinsic motivation
    COMPETITION = "competition" # Resource competition
    COOPERATION = "cooperation" # Team-based rewards
    MIXED = "mixed"            # Combination of above


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    # World dimensions
    world_width: int = 800
    world_height: int = 600
    
    # Agent properties
    agent_radius: int = 10
    agent_speed: float = 2.0
    agent_vision_range: float = 100.0
    agent_attack_range: float = 30.0
    
    # Sensor system (matching existing projects)
    num_sensor_rays: int = 30
    sensor_range: float = 100.0
    observation_dim: int = 152  # 30 rays * 5 values + 2 proprioception
    action_dim: int = 4  # [move_x, move_y, speed, attack_prob]
    
    # Resources
    max_food_items: int = 20
    max_poison_items: int = 10
    food_spawn_rate: float = 0.1
    poison_spawn_rate: float = 0.05
    food_reward: float = 1.0
    poison_penalty: float = -1.0
    
    # Physics
    friction: float = 0.95
    wall_bounce: bool = True
    agent_collision: bool = True
    
    # Rendering
    fps: int = 60
    render_sensors: bool = True
    render_trails: bool = False


@dataclass
class CuriousAgentConfig:
    """Configuration for PPO + Curiosity agents"""
    # PPO parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Curiosity parameters
    curiosity_weight: float = 0.1
    curiosity_lr: float = 1e-3
    feature_dim: int = 64
    
    # Training parameters
    buffer_size: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 2


@dataclass
class CompetitiveAgentConfig:
    """Configuration for A3C Competitive agents"""
    # A3C parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    max_steps: int = 5
    
    # Trust region parameters
    trust_region_coef: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Multi-agent parameters
    sharing_interval: int = 1000
    num_workers: int = 4
    
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 2


@dataclass
class HybridAgentConfig:
    """Configuration for Hybrid agents"""
    # Strategy switching
    strategy_switch_interval: int = 1000
    performance_window: int = 100
    switch_threshold: float = 0.1
    
    # Inherit from both base configs
    curious_config: CuriousAgentConfig = field(default_factory=CuriousAgentConfig)
    competitive_config: CompetitiveAgentConfig = field(default_factory=CompetitiveAgentConfig)
    
    # Hybrid-specific parameters
    strategy_momentum: float = 0.9
    exploration_bonus: float = 0.05


@dataclass
class AdaptiveAgentConfig:
    """Configuration for Adaptive agents"""
    # Meta-learning parameters
    meta_learning_rate: float = 1e-3
    adaptation_steps: int = 5
    meta_batch_size: int = 32
    
    # Strategy evaluation
    evaluation_episodes: int = 10
    strategy_update_interval: int = 500
    
    # Base configurations
    base_configs: Dict[str, object] = field(default_factory=lambda: {
        'curious': CuriousAgentConfig(),
        'competitive': CompetitiveAgentConfig()
    })


@dataclass
class TrainingConfig:
    """Training configuration"""
    # General training
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 100
    
    # Population settings
    total_agents: int = 8
    agent_composition: Dict[AgentType, int] = field(default_factory=lambda: {
        AgentType.CURIOUS: 2,
        AgentType.COMPETITIVE: 2,
        AgentType.HYBRID: 2,
        AgentType.ADAPTIVE: 2
    })
    
    # Reward system
    reward_type: RewardType = RewardType.MIXED
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'curiosity': 0.3,
        'competition': 0.4,
        'cooperation': 0.2,
        'survival': 0.1
    })
    
    # Experiment tracking
    experiment_name: str = "hybrid_arena_experiment"
    save_models: bool = True
    save_logs: bool = True
    save_videos: bool = False


@dataclass
class WebConfig:
    """Web interface configuration"""
    host: str = "0.0.0.0"
    port: int = 7000
    debug: bool = False
    
    # Interface settings
    update_frequency: int = 10  # Hz
    max_clients: int = 10
    
    # Visualization
    canvas_width: int = 800
    canvas_height: int = 600
    show_sensors: bool = True
    show_trails: bool = False
    show_stats: bool = True
    
    # Real-time controls
    allow_parameter_changes: bool = True
    allow_agent_switching: bool = True
    allow_population_changes: bool = True


@dataclass
class HybridArenaConfig:
    """Main configuration class combining all components"""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    curious_agent: CuriousAgentConfig = field(default_factory=CuriousAgentConfig)
    competitive_agent: CompetitiveAgentConfig = field(default_factory=CompetitiveAgentConfig)
    hybrid_agent: HybridAgentConfig = field(default_factory=HybridAgentConfig)
    adaptive_agent: AdaptiveAgentConfig = field(default_factory=AdaptiveAgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    web: WebConfig = field(default_factory=WebConfig)
    
    # Global settings
    device: str = "auto"  # "auto", "cpu", "cuda"
    seed: Optional[int] = None
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Validate agent composition
        total_composed = sum(self.training.agent_composition.values())
        if total_composed != self.training.total_agents:
            print(f"Warning: Agent composition ({total_composed}) doesn't match total_agents ({self.training.total_agents})")
            self.training.total_agents = total_composed
        
        # Set device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HybridArenaConfig':
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclass instances
        # This is a simplified version - full implementation would handle nested conversion
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        import yaml
        from dataclasses import asdict
        
        config_dict = asdict(self)
        # Convert enums to strings
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj
        
        config_dict = convert_enums(config_dict)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_agent_config(self, agent_type: AgentType):
        """Get configuration for specific agent type"""
        config_map = {
            AgentType.CURIOUS: self.curious_agent,
            AgentType.COMPETITIVE: self.competitive_agent,
            AgentType.HYBRID: self.hybrid_agent,
            AgentType.ADAPTIVE: self.adaptive_agent
        }
        return config_map.get(agent_type)


# Default configuration instance
DEFAULT_CONFIG = HybridArenaConfig()


def load_config(config_path: Optional[str] = None) -> HybridArenaConfig:
    """Load configuration from file or return default"""
    if config_path and os.path.exists(config_path):
        return HybridArenaConfig.from_yaml(config_path)
    return DEFAULT_CONFIG


if __name__ == "__main__":
    # Test configuration
    config = HybridArenaConfig()
    print("Configuration loaded successfully!")
    print(f"Total agents: {config.training.total_agents}")
    print(f"Agent composition: {config.training.agent_composition}")
    print(f"Environment size: {config.environment.world_width}x{config.environment.world_height}")
    print(f"Web interface: http://localhost:{config.web.port}")

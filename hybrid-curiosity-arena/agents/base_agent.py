"""
Base agent interface for the Hybrid Multi-Agent Curiosity Arena
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from config.base_config import AgentType, HybridArenaConfig


@dataclass
class AgentObservation:
    """Standardized observation format for all agents"""
    # Raw sensor data (152D)
    sensor_data: np.ndarray  # Shape: (152,)
    
    # Parsed components for convenience
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    sensor_rays: np.ndarray  # Shape: (30, 5) - [distance, food, poison, vel_x, vel_y]
    
    # Multi-agent specific
    nearby_agents: List[Dict[str, Any]]  # Other agents in vision range
    agent_id: int
    agent_type: AgentType
    
    # Environment state
    world_bounds: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    timestep: int
    
    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert observation to PyTorch tensor"""
        return torch.FloatTensor(self.sensor_data).to(device)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'sensor_data': self.sensor_data.tolist(),
            'position': self.position,
            'velocity': self.velocity,
            'nearby_agents': self.nearby_agents,
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'world_bounds': self.world_bounds,
            'timestep': self.timestep
        }


@dataclass
class AgentAction:
    """Standardized action format for all agents"""
    # Core action (4D continuous)
    move_direction: Tuple[float, float]  # [-1, 1] normalized
    move_speed: float  # [0, 1] normalized
    attack_probability: float  # [0, 1] probability of attacking
    
    # Additional metadata
    agent_id: int
    confidence: float = 1.0  # How confident the agent is in this action
    strategy_used: Optional[str] = None  # Which strategy was used
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for environment"""
        return np.array([
            self.move_direction[0],
            self.move_direction[1], 
            self.move_speed,
            self.attack_probability
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'move_direction': self.move_direction,
            'move_speed': self.move_speed,
            'attack_probability': self.attack_probability,
            'agent_id': self.agent_id,
            'confidence': self.confidence,
            'strategy_used': self.strategy_used
        }


@dataclass
class AgentReward:
    """Comprehensive reward structure"""
    # Core reward components
    extrinsic_reward: float = 0.0  # Environment rewards (food, poison, etc.)
    intrinsic_reward: float = 0.0  # Curiosity, exploration bonuses
    competitive_reward: float = 0.0  # Competition-based rewards
    cooperative_reward: float = 0.0  # Cooperation bonuses
    survival_reward: float = 0.0  # Basic survival incentive
    
    # Metadata
    total_reward: float = 0.0
    reward_breakdown: Dict[str, float] = None
    
    def __post_init__(self):
        """Calculate total reward and breakdown"""
        self.total_reward = (
            self.extrinsic_reward + 
            self.intrinsic_reward + 
            self.competitive_reward + 
            self.cooperative_reward + 
            self.survival_reward
        )
        
        self.reward_breakdown = {
            'extrinsic': self.extrinsic_reward,
            'intrinsic': self.intrinsic_reward,
            'competitive': self.competitive_reward,
            'cooperative': self.cooperative_reward,
            'survival': self.survival_reward,
            'total': self.total_reward
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the hybrid arena.
    
    This provides a unified interface that all agent types must implement,
    allowing different learning algorithms to coexist and interact.
    """
    
    def __init__(self, 
                 agent_id: int,
                 agent_type: AgentType,
                 config: HybridArenaConfig,
                 device: str = "cpu"):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (curious, competitive, etc.)
            config: Global configuration
            device: PyTorch device
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.device = device
        
        # Agent state
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.health = 100.0
        self.energy = 100.0
        self.alive = True
        
        # Performance tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.total_episodes = 0
        self.lifetime_reward = 0.0
        
        # Learning state
        self.training = True
        self.last_observation = None
        self.last_action = None
        self.last_reward = None
        
        # Strategy tracking (for hybrid/adaptive agents)
        self.current_strategy = None
        self.strategy_history = []
        self.performance_history = []
    
    @abstractmethod
    def select_action(self, observation: AgentObservation, training: bool = True) -> AgentAction:
        """
        Select action based on current observation
        
        Args:
            observation: Current observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, 
               observation: AgentObservation,
               action: AgentAction, 
               reward: AgentReward,
               next_observation: Optional[AgentObservation],
               done: bool) -> Dict[str, float]:
        """
        Update agent based on experience
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation (None if episode ended)
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def reset_episode(self):
        """Reset agent for new episode"""
        pass
    
    def step(self, observation: AgentObservation) -> AgentAction:
        """
        Main step function called by environment
        
        Args:
            observation: Current observation
            
        Returns:
            Action to take
        """
        # Store observation for learning
        self.last_observation = observation
        
        # Select action
        action = self.select_action(observation, self.training)
        self.last_action = action
        
        # Update episode tracking
        self.episode_steps += 1
        
        return action
    
    def receive_reward(self, reward: AgentReward):
        """
        Receive reward from environment
        
        Args:
            reward: Reward structure
        """
        self.last_reward = reward
        self.episode_reward += reward.total_reward
        self.lifetime_reward += reward.total_reward
    
    def end_episode(self, final_observation: Optional[AgentObservation] = None):
        """
        End current episode
        
        Args:
            final_observation: Final observation of episode
        """
        # Update learning if we have complete experience
        if (self.last_observation is not None and 
            self.last_action is not None and 
            self.last_reward is not None):
            
            metrics = self.update(
                self.last_observation,
                self.last_action,
                self.last_reward,
                final_observation,
                done=True
            )
        
        # Reset episode state
        self.total_episodes += 1
        self.performance_history.append(self.episode_reward)
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # Reset for new episode
        self.reset_episode()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        recent_performance = self.performance_history[-100:] if self.performance_history else [0.0]
        
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'alive': self.alive,
            'health': self.health,
            'energy': self.energy,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps,
            'total_episodes': self.total_episodes,
            'lifetime_reward': self.lifetime_reward,
            'avg_recent_reward': np.mean(recent_performance),
            'current_strategy': self.current_strategy,
            'training': self.training
        }
    
    def set_training(self, training: bool):
        """Set training mode"""
        self.training = training
    
    def save(self, filepath: str):
        """Save agent state"""
        state = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'config': self.config,
            'performance_history': self.performance_history,
            'strategy_history': self.strategy_history,
            'total_episodes': self.total_episodes,
            'lifetime_reward': self.lifetime_reward
        }
        torch.save(state, filepath)
    
    def load(self, filepath: str):
        """Load agent state"""
        state = torch.load(filepath, map_location=self.device)
        self.performance_history = state.get('performance_history', [])
        self.strategy_history = state.get('strategy_history', [])
        self.total_episodes = state.get('total_episodes', 0)
        self.lifetime_reward = state.get('lifetime_reward', 0.0)
    
    def clone(self, new_agent_id: int) -> 'BaseAgent':
        """Create a copy of this agent with new ID"""
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement clone method")
    
    def get_action_distribution(self, observation: AgentObservation) -> Dict[str, float]:
        """
        Get action probability distribution (for analysis)
        
        Args:
            observation: Current observation
            
        Returns:
            Dictionary of action probabilities
        """
        # Default implementation - subclasses can override
        return {'uniform': 1.0}
    
    def get_value_estimate(self, observation: AgentObservation) -> float:
        """
        Get value estimate for current state (for analysis)
        
        Args:
            observation: Current observation
            
        Returns:
            Value estimate
        """
        # Default implementation - subclasses can override
        return 0.0
    
    def switch_strategy(self, new_strategy: str):
        """
        Switch to new strategy (for hybrid/adaptive agents)
        
        Args:
            new_strategy: Name of new strategy
        """
        if self.current_strategy != new_strategy:
            self.strategy_history.append({
                'timestep': self.episode_steps,
                'old_strategy': self.current_strategy,
                'new_strategy': new_strategy,
                'performance': self.episode_reward
            })
            self.current_strategy = new_strategy


if __name__ == "__main__":
    # Test base classes
    from config.base_config import HybridArenaConfig
    
    config = HybridArenaConfig()
    
    # Test observation
    obs = AgentObservation(
        sensor_data=np.random.randn(152),
        position=(100.0, 200.0),
        velocity=(1.0, -0.5),
        sensor_rays=np.random.randn(30, 5),
        nearby_agents=[],
        agent_id=0,
        agent_type=AgentType.CURIOUS,
        world_bounds=(0, 0, 800, 600),
        timestep=100
    )
    
    # Test action
    action = AgentAction(
        move_direction=(0.5, -0.3),
        move_speed=0.8,
        attack_probability=0.1,
        agent_id=0,
        strategy_used="curious"
    )
    
    # Test reward
    reward = AgentReward(
        extrinsic_reward=1.0,
        intrinsic_reward=0.5,
        competitive_reward=-0.2,
        survival_reward=0.1
    )
    
    print("Base classes created successfully!")
    print(f"Observation shape: {obs.sensor_data.shape}")
    print(f"Action array: {action.to_array()}")
    print(f"Total reward: {reward.total_reward}")

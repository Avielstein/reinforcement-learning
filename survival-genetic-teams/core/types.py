"""
Type definitions and data structures for the Multi-Agent Genetic Team Survival System
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

class AgentState(Enum):
    """Possible states for an agent"""
    ALIVE = "alive"
    DEAD = "dead"

class TeamStatus(Enum):
    """Possible statuses for a team"""
    ACTIVE = "active"
    ELIMINATED = "eliminated"
    SPLITTING = "splitting"

class ActionType(Enum):
    """Types of actions an agent can take"""
    MOVE = "move"
    ATTACK = "attack"
    IDLE = "idle"

@dataclass
class Position:
    """2D position with utility methods"""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def direction_to(self, other: 'Position') -> Tuple[float, float]:
        """Get normalized direction vector to another position"""
        dx = other.x - self.x
        dy = other.y - self.y
        dist = self.distance_to(other)
        if dist == 0:
            return (0.0, 0.0)
        return (dx / dist, dy / dist)
    
    def add(self, dx: float, dy: float) -> 'Position':
        """Return new position offset by dx, dy"""
        return Position(self.x + dx, self.y + dy)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple for easy unpacking"""
        return (self.x, self.y)

@dataclass
class AgentAction:
    """Action taken by an agent"""
    action_type: ActionType
    move_direction: Tuple[float, float] = (0.0, 0.0)  # Normalized direction
    move_speed: float = 1.0  # Speed multiplier
    target_id: Optional[int] = None  # For attack actions
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for neural network processing"""
        return np.array([
            self.move_direction[0],
            self.move_direction[1],
            self.move_speed,
            1.0 if self.action_type == ActionType.ATTACK else 0.0
        ])

@dataclass
class AgentObservation:
    """What an agent can observe about its environment"""
    position: Position
    health: float
    team_id: int
    nearby_agents: List[Dict]  # List of {id, position, team_id, health, distance}
    world_bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    
    def to_array(self) -> np.ndarray:
        """Convert observation to neural network input"""
        # Base observation: position, health, normalized
        obs = [
            self.position.x / self.world_bounds[2],  # Normalized x
            self.position.y / self.world_bounds[3],  # Normalized y
            self.health / 100.0,  # Normalized health
        ]
        
        # Add information about nearby agents (up to 10 closest)
        nearby_sorted = sorted(self.nearby_agents, key=lambda a: a['distance'])[:10]
        
        for i in range(10):  # Fixed size for neural network
            if i < len(nearby_sorted):
                agent = nearby_sorted[i]
                obs.extend([
                    (agent['position'].x - self.position.x) / 100.0,  # Relative x
                    (agent['position'].y - self.position.y) / 100.0,  # Relative y
                    agent['health'] / 100.0,  # Their health
                    1.0 if agent['team_id'] == self.team_id else -1.0,  # Friend/foe
                    agent['distance'] / 100.0  # Distance
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # Padding
        
        return np.array(obs, dtype=np.float32)

@dataclass
class EpisodeResult:
    """Results from a single episode"""
    episode_id: int
    total_steps: int
    team_survivors: Dict[int, int]  # team_id -> number of survivors
    team_initial_sizes: Dict[int, int]  # team_id -> initial size
    agent_lifetimes: Dict[int, int]  # agent_id -> steps survived
    team_eliminations: List[int]  # team_ids that were completely eliminated
    
    def get_survival_rate(self, team_id: int) -> float:
        """Get survival rate for a team"""
        if team_id not in self.team_initial_sizes:
            return 0.0
        initial = self.team_initial_sizes[team_id]
        survivors = self.team_survivors.get(team_id, 0)
        return survivors / initial if initial > 0 else 0.0

@dataclass
class TeamStats:
    """Statistics for a team over multiple episodes"""
    team_id: int
    generation: int
    current_size: int
    survival_rates: List[float]  # Last N episodes
    average_survival_rate: float
    generations_since_growth: int
    generations_since_decline: int
    total_eliminations: int
    
    def update_with_episode(self, episode_result: EpisodeResult):
        """Update stats with new episode result"""
        survival_rate = episode_result.get_survival_rate(self.team_id)
        self.survival_rates.append(survival_rate)
        
        # Keep only last 10 episodes for moving average
        if len(self.survival_rates) > 10:
            self.survival_rates.pop(0)
        
        self.average_survival_rate = sum(self.survival_rates) / len(self.survival_rates)
        
        if self.team_id in episode_result.team_eliminations:
            self.total_eliminations += 1

@dataclass
class SimulationState:
    """Current state of the entire simulation"""
    episode: int
    step: int
    teams: Dict[int, 'Team']  # Forward reference
    agents: Dict[int, 'SurvivalAgent']  # Forward reference
    team_stats: Dict[int, TeamStats]
    is_running: bool
    total_agents_alive: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for web interface"""
        return {
            'episode': self.episode,
            'step': self.step,
            'is_running': self.is_running,
            'total_agents_alive': self.total_agents_alive,
            'teams': {
                team_id: {
                    'id': team_id,
                    'size': len(team.agents),
                    'color': team.color,
                    'generation': team.generation,
                    'alive_count': sum(1 for agent in team.agents if agent.is_alive())
                }
                for team_id, team in self.teams.items()
            },
            'agents': {
                agent_id: {
                    'id': agent_id,
                    'team_id': agent.team_id,
                    'position': agent.position.to_tuple(),
                    'health': agent.health,
                    'alive': agent.is_alive()
                }
                for agent_id, agent in self.agents.items()
            }
        }

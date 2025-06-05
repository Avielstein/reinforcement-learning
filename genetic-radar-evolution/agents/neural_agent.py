"""
Neural agent that uses evolved policy networks for decision making
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from .policy_network import PolicyNetwork
# from .behavior_analyzer import BehaviorAnalyzer

class NeuralAgent:
    """Agent that uses neural networks for tactical decisions"""
    
    def __init__(self, agent_id: str, species_id: str, policy_network: PolicyNetwork):
        self.id = agent_id
        self.species_id = species_id
        self.policy = policy_network
        
        # Physical state
        self.x: float = 0.0
        self.y: float = 0.0
        self.heading: float = 0.0
        self.velocity: np.ndarray = np.zeros(2)
        
        # Combat state
        self.health: float = 100.0
        self.max_health: float = 100.0
        self.energy: float = 100.0
        self.max_energy: float = 100.0
        self.reload_time_remaining: float = 0.0
        
        # Tactical state
        self.detected_enemies: List['NeuralAgent'] = []
        self.detected_allies: List['NeuralAgent'] = []
        self.current_target: Optional['NeuralAgent'] = None
        
        # Performance tracking
        self.kills: int = 0
        self.damage_dealt: float = 0.0
        self.damage_received: float = 0.0
        self.survival_time: float = 0.0
        self.shots_fired: int = 0
        self.shots_hit: int = 0
        
        # Behavior tracking
        self.action_history: List[Dict[str, float]] = []
        self.position_history: List[Tuple[float, float]] = []
        
        # Combat parameters (can be influenced by evolution)
        self.base_speed: float = 50.0
        self.radar_range: float = 250.0
        self.firing_range: float = 180.0
        self.reload_time: float = 2.0
        self.accuracy: float = 0.8
    
    def get_observation(self, all_agents: List['NeuralAgent'], map_size: float) -> np.ndarray:
        """Create observation vector for neural network"""
        obs = np.zeros(12)
        
        # Self state (normalized)
        obs[0] = self.x / map_size
        obs[1] = self.y / map_size
        obs[2] = self.health / self.max_health
        obs[3] = self.energy / self.max_energy
        obs[4] = np.cos(self.heading)
        obs[5] = np.sin(self.heading)
        
        # Nearest enemy information
        nearest_enemy = self._find_nearest_enemy(all_agents)
        if nearest_enemy:
            distance = self._distance_to(nearest_enemy)
            obs[6] = min(distance / self.radar_range, 1.0)
            obs[7] = (nearest_enemy.x - self.x) / self.radar_range
            obs[8] = (nearest_enemy.y - self.y) / self.radar_range
            obs[9] = nearest_enemy.health / nearest_enemy.max_health
        
        # Tactical situation
        obs[10] = min(len(self.detected_enemies) / 5.0, 1.0)  # Enemy count (max 5)
        obs[11] = min(len(self.detected_allies) / 5.0, 1.0)   # Ally count (max 5)
        
        return obs
    
    def decide_action(self, all_agents: List['NeuralAgent'], map_size: float) -> Dict[str, Any]:
        """Use neural network to decide actions"""
        observation = self.get_observation(all_agents, map_size)
        action = self.policy.get_action(observation)
        
        # Store for behavior analysis
        self.action_history.append(action.copy())
        if len(self.action_history) > 100:  # Keep last 100 actions
            self.action_history.pop(0)
        
        return action
    
    def update(self, dt: float, all_agents: List['NeuralAgent'], map_size: float) -> None:
        """Update agent state"""
        self.survival_time += dt
        
        # Update reload timer
        if self.reload_time_remaining > 0:
            self.reload_time_remaining -= dt
        
        # Regenerate energy
        self.energy = min(self.max_energy, self.energy + dt * 20)
        
        # Update detection lists
        self._update_detection(all_agents)
        
        # Get action from neural network
        action = self.decide_action(all_agents, map_size)
        
        # Apply movement
        move_speed = self.base_speed * dt
        self.x += action['move_x'] * move_speed
        self.y += action['move_y'] * move_speed
        
        # Keep within bounds
        self.x = np.clip(self.x, 0, map_size)
        self.y = np.clip(self.y, 0, map_size)
        
        # Update heading based on movement
        if abs(action['move_x']) > 0.1 or abs(action['move_y']) > 0.1:
            self.heading = np.arctan2(action['move_y'], action['move_x'])
        
        # Update position history
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > 200:  # Keep last 200 positions
            self.position_history.pop(0)
        
        # Combat decision
        if (action['should_fire'] > 0.5 and 
            self.reload_time_remaining <= 0 and 
            self.detected_enemies):
            
            target = self._select_target(action['target_preference'])
            if target and self._distance_to(target) <= self.firing_range:
                return {'fire_at': target}
        
        return {}
    
    def _update_detection(self, all_agents: List['NeuralAgent']) -> None:
        """Update lists of detected enemies and allies"""
        self.detected_enemies = []
        self.detected_allies = []
        
        for agent in all_agents:
            if agent.id == self.id or agent.health <= 0:
                continue
            
            distance = self._distance_to(agent)
            if distance <= self.radar_range:
                if agent.species_id == self.species_id:
                    self.detected_allies.append(agent)
                else:
                    self.detected_enemies.append(agent)
    
    def _find_nearest_enemy(self, all_agents: List['NeuralAgent']) -> Optional['NeuralAgent']:
        """Find nearest enemy within radar range"""
        nearest = None
        min_distance = float('inf')
        
        for agent in all_agents:
            if (agent.species_id != self.species_id and 
                agent.health > 0):
                distance = self._distance_to(agent)
                if distance < min_distance and distance <= self.radar_range:
                    min_distance = distance
                    nearest = agent
        
        return nearest
    
    def _select_target(self, target_preference: float) -> Optional['NeuralAgent']:
        """Select target based on neural network preference"""
        if not self.detected_enemies:
            return None
        
        if target_preference > 0.5:
            # Prefer strong targets (high health)
            return max(self.detected_enemies, key=lambda e: e.health)
        else:
            # Prefer weak targets (low health)
            return min(self.detected_enemies, key=lambda e: e.health)
    
    def _distance_to(self, other: 'NeuralAgent') -> float:
        """Calculate distance to another agent"""
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)
    
    def take_damage(self, damage: float, attacker_id: str) -> bool:
        """Take damage and return True if killed"""
        self.health -= damage
        self.damage_received += damage
        
        if self.health <= 0:
            self.health = 0
            return True
        return False
    
    def fire_projectile(self, target: 'NeuralAgent') -> Dict[str, Any]:
        """Fire projectile at target"""
        if self.reload_time_remaining > 0:
            return {}
        
        self.shots_fired += 1
        self.reload_time_remaining = self.reload_time
        
        # Calculate projectile trajectory with prediction
        distance = self._distance_to(target)
        projectile_speed = 300.0
        time_to_impact = distance / projectile_speed
        
        # Predict target position
        predicted_x = target.x + target.velocity[0] * time_to_impact
        predicted_y = target.y + target.velocity[1] * time_to_impact
        
        # Add accuracy variation
        accuracy_error = (1.0 - self.accuracy) * 0.3
        predicted_x += np.random.normal(0, accuracy_error * distance)
        predicted_y += np.random.normal(0, accuracy_error * distance)
        
        angle = np.arctan2(predicted_y - self.y, predicted_x - self.x)
        
        return {
            'type': 'projectile',
            'x': self.x,
            'y': self.y,
            'angle': angle,
            'speed': projectile_speed,
            'damage': 25 + np.random.uniform(-5, 5),
            'shooter_id': self.id,
            'species_id': self.species_id
        }
    
    def get_behavioral_metrics(self) -> Dict[str, float]:
        """Get metrics for behavior analysis"""
        if not self.action_history:
            return {}
        
        recent_actions = self.action_history[-50:]  # Last 50 actions
        
        metrics = {
            'avg_aggression': np.mean([a['should_fire'] for a in recent_actions]),
            'avg_exploration': np.mean([a['exploration'] for a in recent_actions]),
            'avg_cooperation': np.mean([a['cooperation'] for a in recent_actions]),
            'movement_variance': np.var([a['move_x']**2 + a['move_y']**2 for a in recent_actions]),
            'target_preference': np.mean([a['target_preference'] for a in recent_actions]),
            'accuracy_ratio': self.shots_hit / max(1, self.shots_fired),
            'survival_time': self.survival_time,
            'kill_death_ratio': self.kills / max(1, 1 if self.health <= 0 else 0),
            'damage_efficiency': self.damage_dealt / max(1, self.damage_received)
        }
        
        return metrics
    
    def get_emergent_behaviors(self) -> List[str]:
        """Identify emergent behaviors from action patterns"""
        behaviors = []
        metrics = self.get_behavioral_metrics()
        
        if not metrics:
            return behaviors
        
        # Analyze behavioral patterns
        if metrics['avg_aggression'] > 0.8:
            behaviors.append('highly_aggressive')
        elif metrics['avg_aggression'] < 0.2:
            behaviors.append('passive')
        
        if metrics['avg_exploration'] > 0.7:
            behaviors.append('explorer')
        elif metrics['avg_exploration'] < 0.3:
            behaviors.append('territorial')
        
        if metrics['avg_cooperation'] > 0.7:
            behaviors.append('team_player')
        elif metrics['avg_cooperation'] < 0.3:
            behaviors.append('lone_wolf')
        
        if metrics['movement_variance'] > 0.5:
            behaviors.append('erratic_movement')
        elif metrics['movement_variance'] < 0.1:
            behaviors.append('steady_movement')
        
        if metrics['accuracy_ratio'] > 0.8:
            behaviors.append('sniper')
        elif metrics['accuracy_ratio'] < 0.3:
            behaviors.append('spray_and_pray')
        
        return behaviors

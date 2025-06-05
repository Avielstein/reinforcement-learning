"""
Individual survival agent with learning and genetic capabilities
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
import random

from core.types import Position, AgentAction, AgentObservation, ActionType, AgentState
from core.config import Config
from agents.policy_net import PolicyNetwork
from agents.experience import ExperienceBuffer

class SurvivalAgent:
    """
    Individual agent that learns to survive in a multi-agent environment
    Uses neural network policy that can evolve genetically
    """
    
    def __init__(self, agent_id: int, team_id: int, config: Config, 
                 policy_network: Optional[PolicyNetwork] = None):
        """
        Initialize survival agent
        
        Args:
            agent_id: Unique identifier for this agent
            team_id: ID of the team this agent belongs to
            config: Global configuration
            policy_network: Pre-trained network, or None to create new one
        """
        self.agent_id = agent_id
        self.team_id = team_id
        self.config = config
        
        # Agent state
        self.position = Position(0.0, 0.0)
        self.health = config.AGENT_HEALTH
        self.max_health = config.AGENT_HEALTH
        self.state = AgentState.ALIVE
        self.age = 0  # Steps survived in current episode
        
        # Neural network policy
        self.policy = policy_network if policy_network else PolicyNetwork()
        
        # Experience tracking
        self.experience_buffer = ExperienceBuffer()
        
        # Learning state
        self.last_observation = None
        self.last_action = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # Performance tracking
        self.lifetime_stats = {
            'episodes_survived': 0,
            'total_kills': 0,
            'total_damage_dealt': 0,
            'total_damage_taken': 0,
            'average_survival_time': 0.0,
            'best_survival_time': 0
        }
    
    def reset_for_episode(self, spawn_position: Position):
        """Reset agent for a new episode"""
        self.position = spawn_position
        self.health = self.max_health
        self.state = AgentState.ALIVE
        self.age = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.last_observation = None
        self.last_action = None
        
        # Start new episode in experience buffer
        self.experience_buffer.start_episode(self.agent_id, self.team_id)
    
    def is_alive(self) -> bool:
        """Check if agent is alive"""
        return self.state == AgentState.ALIVE and self.health > 0
    
    def take_damage(self, damage: float) -> bool:
        """
        Apply damage to agent
        
        Args:
            damage: Amount of damage to apply
            
        Returns:
            True if agent died from this damage
        """
        if not self.is_alive():
            return False
        
        self.health -= damage
        self.lifetime_stats['total_damage_taken'] += damage
        
        if self.health <= 0:
            self.health = 0
            self.state = AgentState.DEAD
            return True
        
        return False
    
    def heal(self, amount: float):
        """Heal the agent"""
        if self.is_alive():
            self.health = min(self.health + amount, self.max_health)
    
    def get_observation(self, nearby_agents: List[Dict]) -> AgentObservation:
        """
        Create observation for this agent
        
        Args:
            nearby_agents: List of nearby agent information
            
        Returns:
            AgentObservation object
        """
        world_bounds = (0, 0, self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT)
        
        return AgentObservation(
            position=self.position,
            health=self.health,
            team_id=self.team_id,
            nearby_agents=nearby_agents,
            world_bounds=world_bounds
        )
    
    def select_action(self, observation: AgentObservation, 
                     add_exploration: bool = True) -> AgentAction:
        """
        Select action based on current observation
        
        Args:
            observation: Current observation
            add_exploration: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        if not self.is_alive():
            return AgentAction(ActionType.IDLE)
        
        # Convert observation to neural network input
        obs_array = observation.to_array()
        
        # Get action from policy network
        noise_scale = 0.1 if add_exploration else 0.0
        action_array = self.policy.get_action(obs_array, add_noise=add_exploration, 
                                            noise_scale=noise_scale)
        
        # Convert to AgentAction
        move_direction = (float(action_array[0]), float(action_array[1]))
        move_speed = float(action_array[2])
        attack_prob = float(action_array[3])
        
        # Determine action type based on attack probability
        action_type = ActionType.ATTACK if attack_prob > 0.5 else ActionType.MOVE
        
        # Find target for attack if attacking
        target_id = None
        if action_type == ActionType.ATTACK and observation.nearby_agents:
            # Target nearest enemy
            enemies = [agent for agent in observation.nearby_agents 
                      if agent['team_id'] != self.team_id]
            if enemies:
                nearest_enemy = min(enemies, key=lambda a: a['distance'])
                if nearest_enemy['distance'] <= self.config.AGENT_ATTACK_RANGE:
                    target_id = nearest_enemy['id']
                else:
                    action_type = ActionType.MOVE  # Too far to attack
        
        return AgentAction(
            action_type=action_type,
            move_direction=move_direction,
            move_speed=move_speed,
            target_id=target_id
        )
    
    def execute_action(self, action: AgentAction) -> Position:
        """
        Execute the given action and return new position
        
        Args:
            action: Action to execute
            
        Returns:
            New position after action
        """
        if not self.is_alive():
            return self.position
        
        new_position = self.position
        
        if action.action_type in [ActionType.MOVE, ActionType.ATTACK]:
            # Calculate movement
            dx = action.move_direction[0] * action.move_speed * self.config.AGENT_SPEED
            dy = action.move_direction[1] * action.move_speed * self.config.AGENT_SPEED
            
            # Apply movement with bounds checking
            new_x = max(0, min(self.config.WORLD_WIDTH, self.position.x + dx))
            new_y = max(0, min(self.config.WORLD_HEIGHT, self.position.y + dy))
            new_position = Position(new_x, new_y)
        
        self.position = new_position
        self.age += 1
        self.episode_steps += 1
        
        return new_position
    
    def calculate_reward(self, observation: AgentObservation, action: AgentAction,
                        damage_dealt: float = 0.0, damage_taken: float = 0.0) -> float:
        """
        Calculate reward for the current step
        
        Args:
            observation: Current observation
            action: Action taken
            damage_dealt: Damage dealt to enemies this step
            damage_taken: Damage taken this step
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Base survival reward
        reward += 2.0
        
        # Health-based reward (encourage staying healthy)
        health_ratio = self.health / self.max_health
        reward += health_ratio * 1.0
        
        # Combat rewards (much higher to encourage fighting)
        reward += damage_dealt * 5.0  # High reward for dealing damage
        reward -= damage_taken * 2.0  # Penalty for taking damage
        
        # Movement reward (encourage active movement)
        if action.action_type == ActionType.MOVE:
            reward += action.move_speed * 0.5  # Reward for moving fast
        
        # Proximity and engagement rewards
        if observation.nearby_agents:
            enemies = [agent for agent in observation.nearby_agents 
                      if agent['team_id'] != self.team_id]
            teammates = [agent for agent in observation.nearby_agents 
                        if agent['team_id'] == self.team_id]
            
            if enemies:
                nearest_enemy_distance = min(agent['distance'] for agent in enemies)
                
                # Strong reward for being close to enemies
                if nearest_enemy_distance < self.config.AGENT_ATTACK_RANGE:
                    reward += 3.0  # Very close to enemy
                elif nearest_enemy_distance < self.config.AGENT_VISION_RANGE * 0.5:
                    reward += 1.5  # Moderately close to enemy
                
                # Reward for attacking when enemies are nearby
                if action.action_type == ActionType.ATTACK and nearest_enemy_distance < self.config.AGENT_ATTACK_RANGE:
                    reward += 2.0
            
            # Small reward for staying near teammates (pack behavior)
            if teammates:
                avg_teammate_distance = sum(agent['distance'] for agent in teammates) / len(teammates)
                if avg_teammate_distance < self.config.AGENT_VISION_RANGE * 0.3:
                    reward += 0.5
        
        # Penalty for being idle
        if action.action_type == ActionType.IDLE:
            reward -= 1.0
        
        # Small penalty for being near world edges
        edge_penalty = 0.0
        margin = 100.0
        if self.position.x < margin or self.position.x > self.config.WORLD_WIDTH - margin:
            edge_penalty += 0.2
        if self.position.y < margin or self.position.y > self.config.WORLD_HEIGHT - margin:
            edge_penalty += 0.2
        reward -= edge_penalty
        
        return reward
    
    def learn_from_experience(self, learning_rate: Optional[float] = None):
        """
        Update policy based on recent experiences
        Simple policy gradient update for genetic evolution
        """
        if not self.is_alive():
            return
        
        lr = learning_rate or self.config.LEARNING_RATE
        
        # Get learning data from experience buffer
        observations, actions, advantages = self.experience_buffer.get_learning_data(recent_episodes=3)
        
        if len(observations) == 0:
            return
        
        # Simple policy gradient update
        self.policy.train()
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations)
        action_tensor = torch.FloatTensor(actions)
        advantage_tensor = torch.FloatTensor(advantages)
        
        # Get policy outputs
        policy_outputs = []
        for obs in obs_tensor:
            output = self.policy.forward(obs)
            policy_outputs.append(output)
        
        policy_tensor = torch.stack(policy_outputs)
        
        # Calculate policy gradient loss
        # Simple MSE loss weighted by advantages
        loss = torch.mean((policy_tensor - action_tensor) ** 2 * advantage_tensor.unsqueeze(1))
        
        # Manual gradient update (simple version)
        with torch.no_grad():
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad
    
    def share_policy_with_team(self, teammate_policies: List[PolicyNetwork], 
                              sharing_strength: float):
        """
        Share and learn from teammate policies
        
        Args:
            teammate_policies: List of teammate policy networks
            sharing_strength: How much to blend with teammate policies
        """
        if not teammate_policies:
            return
        
        # Average teammate policies
        avg_weights = None
        for teammate_policy in teammate_policies:
            teammate_weights = teammate_policy.get_weights_vector()
            if avg_weights is None:
                avg_weights = teammate_weights.copy()
            else:
                avg_weights += teammate_weights
        
        if avg_weights is not None:
            avg_weights /= len(teammate_policies)
            
            # Blend with own policy
            own_weights = self.policy.get_weights_vector()
            blended_weights = (1 - sharing_strength) * own_weights + sharing_strength * avg_weights
            
            self.policy.set_weights_vector(blended_weights)
    
    def record_experience(self, observation: AgentObservation, action: AgentAction,
                         reward: float, next_observation: Optional[AgentObservation],
                         done: bool):
        """Record experience for learning"""
        obs_array = observation.to_array()
        action_array = action.to_array()
        next_obs_array = next_observation.to_array() if next_observation else None
        
        self.experience_buffer.add_experience(
            observation=obs_array,
            action=action_array,
            reward=reward,
            next_observation=next_obs_array,
            done=done,
            step=self.episode_steps
        )
        
        self.episode_reward += reward
    
    def end_episode(self, survived: bool):
        """End current episode and update statistics"""
        self.experience_buffer.end_episode()
        
        # Update lifetime statistics
        if survived:
            self.lifetime_stats['episodes_survived'] += 1
        
        # Update survival time statistics
        survival_time = self.age
        self.lifetime_stats['best_survival_time'] = max(
            self.lifetime_stats['best_survival_time'], survival_time
        )
        
        # Update average survival time
        total_episodes = self.lifetime_stats['episodes_survived'] + 1
        current_avg = self.lifetime_stats['average_survival_time']
        self.lifetime_stats['average_survival_time'] = (
            (current_avg * (total_episodes - 1) + survival_time) / total_episodes
        )
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for this agent"""
        buffer_metrics = self.experience_buffer.calculate_performance_metrics()
        
        return {
            'agent_id': self.agent_id,
            'team_id': self.team_id,
            'current_health': self.health,
            'is_alive': self.is_alive(),
            'age': self.age,
            'episode_reward': self.episode_reward,
            'lifetime_stats': self.lifetime_stats.copy(),
            'recent_performance': buffer_metrics
        }
    
    def clone(self, new_agent_id: int) -> 'SurvivalAgent':
        """Create a clone of this agent with a new ID"""
        cloned_policy = self.policy.copy()
        clone = SurvivalAgent(new_agent_id, self.team_id, self.config, cloned_policy)
        
        # Copy some statistics but reset episode-specific state
        clone.lifetime_stats = self.lifetime_stats.copy()
        
        return clone
    
    def mutate_policy(self, mutation_rate: Optional[float] = None, 
                     mutation_strength: Optional[float] = None):
        """Apply mutations to the agent's policy"""
        mut_rate = mutation_rate or self.config.MUTATION_RATE
        mut_strength = mutation_strength or self.config.MUTATION_STRENGTH
        
        self.policy.mutate(mut_rate, mut_strength)
    
    def save_policy(self, filepath: str):
        """Save agent's policy to file"""
        self.policy.save(filepath)
    
    def load_policy(self, filepath: str):
        """Load agent's policy from file"""
        self.policy.load(filepath)

"""
Team management for genetic evolution of agent groups
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from core.types import Position, TeamStatus, EpisodeResult
from core.config import Config
from agents.survival_agent import SurvivalAgent
from agents.policy_net import PolicyNetwork
from agents.experience import TeamExperienceManager

@dataclass
class TeamPerformance:
    """Track team performance over episodes"""
    survival_rates: List[float]
    average_survival_rate: float
    episodes_since_growth: int
    episodes_since_decline: int
    total_eliminations: int
    
    def update(self, survival_rate: float, config: Config):
        """Update performance with new episode result"""
        self.survival_rates.append(survival_rate)
        
        # Keep only recent history
        if len(self.survival_rates) > 10:
            self.survival_rates.pop(0)
        
        self.average_survival_rate = sum(self.survival_rates) / len(self.survival_rates)
        
        # Update growth/decline counters
        if survival_rate >= config.GROWTH_THRESHOLD:
            self.episodes_since_growth = 0
            self.episodes_since_decline += 1
        elif survival_rate <= config.SURVIVAL_THRESHOLD:
            self.episodes_since_decline = 0
            self.episodes_since_growth += 1
        else:
            self.episodes_since_growth += 1
            self.episodes_since_decline += 1
        
        if survival_rate == 0.0:
            self.total_eliminations += 1

class Team:
    """
    Manages a team of agents that evolve together through genetic algorithms
    """
    
    def __init__(self, team_id: int, config: Config, initial_size: Optional[int] = None):
        """
        Initialize team
        
        Args:
            team_id: Unique identifier for this team
            config: Global configuration
            initial_size: Initial number of agents (uses config default if None)
        """
        self.team_id = team_id
        self.config = config
        self.color = config.get_team_color(team_id)
        
        # Team state
        self.status = TeamStatus.ACTIVE
        self.generation = 0
        self.agents: List[SurvivalAgent] = []
        
        # Performance tracking
        self.performance = TeamPerformance(
            survival_rates=[],
            average_survival_rate=0.0,
            episodes_since_growth=0,
            episodes_since_decline=0,
            total_eliminations=0
        )
        
        # Experience management
        self.experience_manager = TeamExperienceManager(team_id)
        
        # Cached diversity score (updated only after episodes)
        self._cached_diversity_score = 0.0
        self._diversity_needs_update = True
        
        # Agent ID counter for this team
        self._next_agent_id = team_id * 1000  # Ensure unique IDs across teams
        
        # Initialize agents
        size = initial_size or config.STARTING_TEAM_SIZE
        self._create_initial_agents(size)
    
    def _create_initial_agents(self, count: int):
        """Create initial agents for the team"""
        for _ in range(count):
            agent = SurvivalAgent(
                agent_id=self._get_next_agent_id(),
                team_id=self.team_id,
                config=self.config
            )
            self.agents.append(agent)
            self.experience_manager.add_agent(agent.agent_id)
    
    def _get_next_agent_id(self) -> int:
        """Get next unique agent ID for this team"""
        agent_id = self._next_agent_id
        self._next_agent_id += 1
        return agent_id
    
    def add_agent(self, agent: Optional[SurvivalAgent] = None) -> SurvivalAgent:
        """
        Add an agent to the team
        
        Args:
            agent: Existing agent to add, or None to create new one
            
        Returns:
            The added agent
        """
        if agent is None:
            # Create new agent with random policy
            agent = SurvivalAgent(
                agent_id=self._get_next_agent_id(),
                team_id=self.team_id,
                config=self.config
            )
        else:
            # Update agent's team ID
            agent.team_id = self.team_id
        
        self.agents.append(agent)
        self.experience_manager.add_agent(agent.agent_id)
        return agent
    
    def remove_agent(self, agent_id: int) -> bool:
        """
        Remove an agent from the team
        
        Args:
            agent_id: ID of agent to remove
            
        Returns:
            True if agent was found and removed
        """
        for i, agent in enumerate(self.agents):
            if agent.agent_id == agent_id:
                self.agents.pop(i)
                self.experience_manager.remove_agent(agent_id)
                return True
        return False
    
    def get_alive_agents(self) -> List[SurvivalAgent]:
        """Get list of currently alive agents"""
        return [agent for agent in self.agents if agent.is_alive()]
    
    def get_spawn_positions(self) -> List[Position]:
        """Generate spawn positions for all agents in the team"""
        positions = []
        
        # Spawn teams closer to center for more interaction
        center_x = self.config.WORLD_WIDTH // 2
        center_y = self.config.WORLD_HEIGHT // 2
        
        # Teams spawn in a circle around center, closer together
        import math
        angle_per_team = 2 * math.pi / max(4, self.config.INITIAL_TEAMS)
        team_angle = self.team_id * angle_per_team
        
        # Smaller radius for closer spawning
        spawn_radius = min(self.config.WORLD_WIDTH, self.config.WORLD_HEIGHT) * 0.25
        
        base_x = center_x + spawn_radius * math.cos(team_angle)
        base_y = center_y + spawn_radius * math.sin(team_angle)
        
        # Ensure within bounds
        base_x = max(100, min(self.config.WORLD_WIDTH - 100, base_x))
        base_y = max(100, min(self.config.WORLD_HEIGHT - 100, base_y))
        
        # Spawn agents in a tight cluster
        for i in range(len(self.agents)):
            # Smaller random offset for tighter clustering
            offset_x = random.uniform(-30, 30)
            offset_y = random.uniform(-30, 30)
            
            x = max(50, min(self.config.WORLD_WIDTH - 50, base_x + offset_x))
            y = max(50, min(self.config.WORLD_HEIGHT - 50, base_y + offset_y))
            
            positions.append(Position(x, y))
        
        return positions
    
    def reset_for_episode(self):
        """Reset all agents for a new episode"""
        spawn_positions = self.get_spawn_positions()
        
        for agent, position in zip(self.agents, spawn_positions):
            agent.reset_for_episode(position)
        
        self.experience_manager.start_episode_for_all()
    
    def update_after_episode(self, episode_result: EpisodeResult):
        """
        Update team state after episode completion
        
        Args:
            episode_result: Results from the completed episode
        """
        # End episode for all agents
        self.experience_manager.end_episode_for_all()
        
        # Calculate survival rate
        survivors = episode_result.team_survivors.get(self.team_id, 0)
        initial_size = episode_result.team_initial_sizes.get(self.team_id, len(self.agents))
        survival_rate = survivors / initial_size if initial_size > 0 else 0.0
        
        # Update performance tracking
        self.performance.update(survival_rate, self.config)
        
        # Update agent statistics
        for agent in self.agents:
            survived = agent.is_alive()
            agent.end_episode(survived)
        
        # Mark diversity cache for update (will be calculated when needed)
        self._update_diversity_cache()
        
        # Determine if team evolution is needed
        self._evolve_team(survival_rate)
    
    def _evolve_team(self, survival_rate: float):
        """
        Evolve the team based on performance
        
        Args:
            survival_rate: Survival rate from last episode
        """
        current_size = len(self.agents)
        survivors = len(self.get_alive_agents())
        
        # Check for team elimination based on minimum survivors requirement
        if survivors < self.config.MIN_SURVIVORS_TO_CONTINUE:
            self.status = TeamStatus.ELIMINATED
            return
        
        # Check for team elimination based on consecutive poor performance
        if survival_rate == 0.0 and self.performance.total_eliminations >= self.config.ELIMINATION_GENERATIONS:
            self.status = TeamStatus.ELIMINATED
            return
        
        # Check for team growth
        if (survival_rate >= self.config.GROWTH_THRESHOLD and 
            current_size < self.config.MAX_TEAM_SIZE):
            self._grow_team()
        
        # Check for team shrinking
        elif (survival_rate <= self.config.SURVIVAL_THRESHOLD and 
              current_size > self.config.MIN_TEAM_SIZE):
            self._shrink_team()
        
        # Check for team splitting
        if current_size >= self.config.SPLIT_THRESHOLD:
            self.status = TeamStatus.SPLITTING
        
        # Apply genetic operations
        self._apply_genetic_operations(survival_rate)
        
        self.generation += 1
    
    def _grow_team(self):
        """Add new agents to the team"""
        # Clone one of the best performing agents
        best_agents = self._get_best_agents(min(3, len(self.agents)))
        
        if best_agents:
            parent = random.choice(best_agents)
            new_agent = parent.clone(self._get_next_agent_id())
            
            # Add some mutation to the new agent
            new_agent.mutate_policy()
            
            self.add_agent(new_agent)
    
    def _shrink_team(self):
        """Remove worst performing agents"""
        if len(self.agents) <= self.config.MIN_TEAM_SIZE:
            return
        
        # Remove the worst performing agent
        worst_agents = self._get_worst_agents(1)
        if worst_agents:
            self.remove_agent(worst_agents[0].agent_id)
    
    def _apply_genetic_operations(self, survival_rate: float):
        """Apply genetic operations to team policies"""
        # Share policies among team members
        self._share_team_policies()
        
        # Apply mutations based on performance
        mutation_rate = self.config.MUTATION_RATE
        if survival_rate < self.config.SURVIVAL_THRESHOLD:
            # Increase mutation rate for poor performance
            mutation_rate *= 2.0
        
        for agent in self.agents:
            if random.random() < 0.5:  # 50% chance to mutate each agent
                agent.mutate_policy(mutation_rate=mutation_rate)
    
    def _share_team_policies(self):
        """Share successful policies among team members"""
        if len(self.agents) < 2:
            return
        
        # Get policies from best performing agents
        best_agents = self._get_best_agents(min(3, len(self.agents)))
        best_policies = [agent.policy for agent in best_agents]
        
        # Share with all team members
        sharing_strength = self.config.POLICY_SHARING_STRENGTH
        for agent in self.agents:
            if agent not in best_agents:  # Don't share with themselves
                agent.share_policy_with_team(best_policies, sharing_strength)
    
    def _get_best_agents(self, count: int) -> List[SurvivalAgent]:
        """Get the best performing agents in the team"""
        if not self.agents:
            return []
        
        # Sort by episode reward (simple performance metric)
        sorted_agents = sorted(self.agents, 
                             key=lambda a: a.episode_reward, 
                             reverse=True)
        
        return sorted_agents[:count]
    
    def _get_worst_agents(self, count: int) -> List[SurvivalAgent]:
        """Get the worst performing agents in the team"""
        if not self.agents:
            return []
        
        # Sort by episode reward (ascending)
        sorted_agents = sorted(self.agents, 
                             key=lambda a: a.episode_reward)
        
        return sorted_agents[:count]
    
    def split_team(self, new_team_id: int) -> 'Team':
        """
        Split this team into two teams
        
        Args:
            new_team_id: ID for the new team
            
        Returns:
            New team created from the split
        """
        if len(self.agents) < 4:  # Need at least 4 agents to split
            return None
        
        # Split agents roughly in half
        split_point = len(self.agents) // 2
        
        # Create new team
        new_team = Team(new_team_id, self.config, initial_size=0)
        new_team.generation = self.generation
        
        # Move half the agents to new team
        agents_to_move = self.agents[split_point:]
        self.agents = self.agents[:split_point]
        
        for agent in agents_to_move:
            self.experience_manager.remove_agent(agent.agent_id)
            new_team.add_agent(agent)
        
        # Apply mutations to differentiate the teams
        for agent in new_team.agents:
            agent.mutate_policy(mutation_rate=self.config.MUTATION_RATE * 2)
        
        self.status = TeamStatus.ACTIVE
        return new_team
    
    def get_team_summary(self) -> Dict:
        """Get summary information about the team"""
        alive_count = len(self.get_alive_agents())
        diversity_score = self.get_diversity_score()
        
        return {
            'team_id': self.team_id,
            'id': self.team_id,  # Add 'id' for consistency with web interface
            'color': self.color,
            'status': self.status.value,
            'generation': self.generation,
            'size': len(self.agents),
            'alive_count': alive_count,
            'survival_rate': alive_count / len(self.agents) if self.agents else 0,
            'average_survival_rate': self.performance.average_survival_rate,
            'episodes_since_growth': self.performance.episodes_since_growth,
            'episodes_since_decline': self.performance.episodes_since_decline,
            'total_eliminations': self.performance.total_eliminations,
            'diversity_score': diversity_score
        }
    
    def get_diversity_score(self) -> float:
        """Get cached diversity score (calculated only after episodes)"""
        if self._diversity_needs_update:
            self._cached_diversity_score = self.experience_manager.calculate_team_diversity()
            self._diversity_needs_update = False
        
        return self._cached_diversity_score
    
    def _update_diversity_cache(self):
        """Mark diversity cache as needing update"""
        self._diversity_needs_update = True
    
    def save_best_policies(self, directory: str):
        """Save policies of best performing agents"""
        best_agents = self._get_best_agents(3)
        
        for i, agent in enumerate(best_agents):
            filepath = f"{directory}/team_{self.team_id}_best_{i}.pt"
            agent.save_policy(filepath)
    
    def is_active(self) -> bool:
        """Check if team is still active"""
        return self.status == TeamStatus.ACTIVE and len(self.agents) > 0
    
    def should_split(self) -> bool:
        """Check if team should be split"""
        return (self.status == TeamStatus.SPLITTING or 
                len(self.agents) >= self.config.SPLIT_THRESHOLD)

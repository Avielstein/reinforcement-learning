"""
Multi-agent survival environment for genetic team evolution
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from core.types import Position, AgentAction, AgentObservation, ActionType, EpisodeResult
from core.config import Config
from agents.survival_agent import SurvivalAgent

class SurvivalEnvironment:
    """
    Environment where multiple teams of agents compete for survival
    """
    
    def __init__(self, config: Config):
        """
        Initialize survival environment
        
        Args:
            config: Global configuration
        """
        self.config = config
        self.step_count = 0
        self.episode_id = 0
        
        # Environment state
        self.agents: Dict[int, SurvivalAgent] = {}
        self.team_agents: Dict[int, List[SurvivalAgent]] = defaultdict(list)
        
        # Combat tracking
        self.combat_log = []
        self.damage_dealt = defaultdict(float)  # agent_id -> total damage dealt
        self.damage_taken = defaultdict(float)  # agent_id -> total damage taken
        
        # Episode tracking
        self.initial_team_sizes = {}
        self.agent_spawn_times = {}
    
    def reset(self, teams: Dict[int, 'Team']) -> Dict[int, AgentObservation]:
        """
        Reset environment for new episode
        
        Args:
            teams: Dictionary of team_id -> Team objects
            
        Returns:
            Initial observations for all agents
        """
        self.step_count = 0
        self.episode_id += 1
        self.agents.clear()
        self.team_agents.clear()
        self.combat_log.clear()
        self.damage_dealt.clear()
        self.damage_taken.clear()
        self.initial_team_sizes.clear()
        self.agent_spawn_times.clear()
        
        # Add all agents from teams
        for team_id, team in teams.items():
            self.initial_team_sizes[team_id] = len(team.agents)
            self.team_agents[team_id] = team.agents.copy()
            
            for agent in team.agents:
                self.agents[agent.agent_id] = agent
                self.agent_spawn_times[agent.agent_id] = 0
        
        # Get initial observations
        observations = {}
        for agent_id, agent in self.agents.items():
            if agent.is_alive():
                nearby_agents = self._get_nearby_agents(agent)
                observations[agent_id] = agent.get_observation(nearby_agents)
        
        return observations
    
    def step(self, actions: Dict[int, AgentAction]) -> Tuple[Dict[int, AgentObservation], 
                                                           Dict[int, float], 
                                                           Dict[int, bool], 
                                                           Dict]:
        """
        Execute one step of the environment
        
        Args:
            actions: Dictionary of agent_id -> AgentAction
            
        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        self.step_count += 1
        
        # Execute all agent actions
        self._execute_actions(actions)
        
        # Handle combat
        self._resolve_combat(actions)
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions)
        
        # Check for episode termination
        dones = self._check_episode_done()
        
        # Get new observations
        observations = {}
        for agent_id, agent in self.agents.items():
            if agent.is_alive():
                nearby_agents = self._get_nearby_agents(agent)
                observations[agent_id] = agent.get_observation(nearby_agents)
        
        # Compile info
        info = {
            'step': self.step_count,
            'alive_agents': len([a for a in self.agents.values() if a.is_alive()]),
            'combat_events': len(self.combat_log),
            'team_alive_counts': {
                team_id: len([a for a in agents if a.is_alive()])
                for team_id, agents in self.team_agents.items()
            }
        }
        
        return observations, rewards, dones, info
    
    def _execute_actions(self, actions: Dict[int, AgentAction]):
        """Execute movement actions for all agents"""
        for agent_id, action in actions.items():
            if agent_id in self.agents and self.agents[agent_id].is_alive():
                agent = self.agents[agent_id]
                agent.execute_action(action)
    
    def _resolve_combat(self, actions: Dict[int, AgentAction]):
        """Resolve combat between agents"""
        # Collect all attack actions
        attack_actions = []
        for agent_id, action in actions.items():
            if (action.action_type == ActionType.ATTACK and 
                agent_id in self.agents and 
                self.agents[agent_id].is_alive() and
                action.target_id is not None):
                attack_actions.append((agent_id, action))
        
        # Process attacks
        for attacker_id, action in attack_actions:
            attacker = self.agents[attacker_id]
            target_id = action.target_id
            
            if target_id not in self.agents:
                continue
            
            target = self.agents[target_id]
            
            # Check if target is still alive and in range
            if not target.is_alive():
                continue
            
            distance = attacker.position.distance_to(target.position)
            if distance > self.config.AGENT_ATTACK_RANGE:
                continue
            
            # Check if they're on different teams
            if attacker.team_id == target.team_id:
                continue
            
            # Calculate damage (with some randomness)
            base_damage = self.config.AGENT_ATTACK_DAMAGE
            damage = base_damage * random.uniform(0.8, 1.2)
            
            # Apply damage
            target_died = target.take_damage(damage)
            
            # Track damage statistics
            self.damage_dealt[attacker_id] += damage
            self.damage_taken[target_id] += damage
            
            # Update attacker's lifetime stats
            attacker.lifetime_stats['total_damage_dealt'] += damage
            if target_died:
                attacker.lifetime_stats['total_kills'] += 1
            
            # Log combat event
            self.combat_log.append({
                'step': self.step_count,
                'attacker_id': attacker_id,
                'attacker_team': attacker.team_id,
                'target_id': target_id,
                'target_team': target.team_id,
                'damage': damage,
                'target_died': target_died,
                'attacker_pos': attacker.position.to_tuple(),
                'target_pos': target.position.to_tuple()
            })
    
    def _get_nearby_agents(self, agent: SurvivalAgent) -> List[Dict]:
        """Get information about agents near the given agent"""
        nearby = []
        
        for other_id, other_agent in self.agents.items():
            if other_id == agent.agent_id or not other_agent.is_alive():
                continue
            
            distance = agent.position.distance_to(other_agent.position)
            
            if distance <= agent.config.AGENT_VISION_RANGE:
                nearby.append({
                    'id': other_id,
                    'position': other_agent.position,
                    'team_id': other_agent.team_id,
                    'health': other_agent.health,
                    'distance': distance
                })
        
        return nearby
    
    def _calculate_rewards(self, actions: Dict[int, AgentAction]) -> Dict[int, float]:
        """Calculate rewards for all agents"""
        rewards = {}
        
        for agent_id, agent in self.agents.items():
            if not agent.is_alive():
                rewards[agent_id] = 0.0
                continue
            
            # Get agent's observation and action
            nearby_agents = self._get_nearby_agents(agent)
            observation = agent.get_observation(nearby_agents)
            action = actions.get(agent_id, AgentAction(ActionType.IDLE))
            
            # Calculate reward using agent's reward function
            damage_dealt = self.damage_dealt.get(agent_id, 0.0)
            damage_taken = self.damage_taken.get(agent_id, 0.0)
            
            reward = agent.calculate_reward(
                observation=observation,
                action=action,
                damage_dealt=damage_dealt,
                damage_taken=damage_taken
            )
            
            rewards[agent_id] = reward
            
            # Reset damage counters for next step
            self.damage_dealt[agent_id] = 0.0
            self.damage_taken[agent_id] = 0.0
        
        return rewards
    
    def _check_episode_done(self) -> Dict[int, bool]:
        """Check if episode is done for each agent"""
        dones = {}
        
        # Episode ends if agent dies or max steps reached
        episode_done = self.step_count >= self.config.EPISODE_LENGTH
        
        for agent_id, agent in self.agents.items():
            agent_done = not agent.is_alive() or episode_done
            dones[agent_id] = agent_done
        
        return dones
    
    def is_episode_complete(self) -> bool:
        """Check if the entire episode is complete"""
        # Episode is complete if max steps reached or only one team remains
        if self.step_count >= self.config.EPISODE_LENGTH:
            return True
        
        # Count teams with alive agents
        teams_with_alive_agents = set()
        for agent in self.agents.values():
            if agent.is_alive():
                teams_with_alive_agents.add(agent.team_id)
        
        # Episode ends when 0 or 1 team remains
        return len(teams_with_alive_agents) <= 1
    
    def get_episode_result(self) -> EpisodeResult:
        """Get results from the completed episode"""
        # Count survivors by team
        team_survivors = defaultdict(int)
        agent_lifetimes = {}
        
        for agent_id, agent in self.agents.items():
            if agent.is_alive():
                team_survivors[agent.team_id] += 1
            
            agent_lifetimes[agent_id] = agent.age
        
        # Find eliminated teams (teams with no survivors)
        team_eliminations = []
        for team_id in self.initial_team_sizes.keys():
            if team_survivors[team_id] == 0:
                team_eliminations.append(team_id)
        
        return EpisodeResult(
            episode_id=self.episode_id,
            total_steps=self.step_count,
            team_survivors=dict(team_survivors),
            team_initial_sizes=self.initial_team_sizes.copy(),
            agent_lifetimes=agent_lifetimes,
            team_eliminations=team_eliminations
        )
    
    def get_combat_statistics(self) -> Dict:
        """Get statistics about combat during the episode"""
        if not self.combat_log:
            return {}
        
        total_combat_events = len(self.combat_log)
        total_damage = sum(event['damage'] for event in self.combat_log)
        total_kills = sum(1 for event in self.combat_log if event['target_died'])
        
        # Combat by team
        team_damage_dealt = defaultdict(float)
        team_damage_taken = defaultdict(float)
        team_kills = defaultdict(int)
        team_deaths = defaultdict(int)
        
        for event in self.combat_log:
            team_damage_dealt[event['attacker_team']] += event['damage']
            team_damage_taken[event['target_team']] += event['damage']
            
            if event['target_died']:
                team_kills[event['attacker_team']] += 1
                team_deaths[event['target_team']] += 1
        
        return {
            'total_combat_events': total_combat_events,
            'total_damage': total_damage,
            'total_kills': total_kills,
            'average_damage_per_attack': total_damage / total_combat_events if total_combat_events > 0 else 0,
            'team_damage_dealt': dict(team_damage_dealt),
            'team_damage_taken': dict(team_damage_taken),
            'team_kills': dict(team_kills),
            'team_deaths': dict(team_deaths)
        }
    
    def get_environment_state(self) -> Dict:
        """Get current state of the environment for visualization"""
        alive_agents = [agent for agent in self.agents.values() if agent.is_alive()]
        
        return {
            'step': self.step_count,
            'episode_id': self.episode_id,
            'total_agents': len(self.agents),
            'alive_agents': len(alive_agents),
            'agent_positions': {
                agent.agent_id: {
                    'position': agent.position.to_tuple(),
                    'team_id': agent.team_id,
                    'health': agent.health,
                    'alive': agent.is_alive()
                }
                for agent in self.agents.values()
            },
            'team_counts': {
                team_id: len([a for a in agents if a.is_alive()])
                for team_id, agents in self.team_agents.items()
            },
            'recent_combat': self.combat_log[-10:] if self.combat_log else []  # Last 10 combat events
        }
    
    def get_team_performance_preview(self) -> Dict:
        """Get a preview of how teams are performing during the episode"""
        team_performance = {}
        
        for team_id, agents in self.team_agents.items():
            alive_count = len([a for a in agents if a.is_alive()])
            total_health = sum(a.health for a in agents if a.is_alive())
            avg_age = sum(a.age for a in agents if a.is_alive()) / alive_count if alive_count > 0 else 0
            
            team_performance[team_id] = {
                'alive_count': alive_count,
                'survival_rate': alive_count / self.initial_team_sizes[team_id],
                'average_health': total_health / alive_count if alive_count > 0 else 0,
                'average_age': avg_age,
                'total_damage_dealt': sum(self.damage_dealt.get(a.agent_id, 0) for a in agents),
                'total_damage_taken': sum(self.damage_taken.get(a.agent_id, 0) for a in agents)
            }
        
        return team_performance
    
    def force_episode_end(self):
        """Force the episode to end (for testing or emergency stops)"""
        self.step_count = self.config.EPISODE_LENGTH
    
    def add_environmental_hazards(self):
        """Add environmental hazards (future feature)"""
        # Placeholder for environmental hazards like:
        # - Shrinking play area
        # - Periodic damage zones
        # - Resource scarcity
        pass
    
    def get_spatial_density_map(self, grid_size: int = 20) -> np.ndarray:
        """Get a spatial density map of agent positions"""
        density_map = np.zeros((grid_size, grid_size))
        
        cell_width = self.config.WORLD_WIDTH / grid_size
        cell_height = self.config.WORLD_HEIGHT / grid_size
        
        for agent in self.agents.values():
            if agent.is_alive():
                grid_x = int(agent.position.x / cell_width)
                grid_y = int(agent.position.y / cell_height)
                
                # Clamp to grid bounds
                grid_x = max(0, min(grid_size - 1, grid_x))
                grid_y = max(0, min(grid_size - 1, grid_y))
                
                density_map[grid_y, grid_x] += 1
        
        return density_map

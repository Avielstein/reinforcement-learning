"""
Population management for multiple teams with genetic evolution
"""

import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from core.types import EpisodeResult, TeamStatus
from core.config import Config
from teams.team import Team

class Population:
    """
    Manages the entire population of teams and handles evolution dynamics
    """
    
    def __init__(self, config: Config):
        """
        Initialize population
        
        Args:
            config: Global configuration
        """
        self.config = config
        self.teams: Dict[int, Team] = {}
        self.next_team_id = 0
        self.generation = 0
        
        # Population statistics
        self.total_agents_created = 0
        self.total_teams_created = 0
        self.eliminated_teams = []
        
        # Initialize starting teams
        self._create_initial_teams()
    
    def _create_initial_teams(self):
        """Create the initial population of teams"""
        for _ in range(self.config.INITIAL_TEAMS):
            team = Team(
                team_id=self._get_next_team_id(),
                config=self.config
            )
            self.teams[team.team_id] = team
            self.total_agents_created += len(team.agents)
    
    def _get_next_team_id(self) -> int:
        """Get next unique team ID"""
        team_id = self.next_team_id
        self.next_team_id += 1
        self.total_teams_created += 1
        return team_id
    
    def get_active_teams(self) -> Dict[int, Team]:
        """Get all currently active teams"""
        return {tid: team for tid, team in self.teams.items() if team.is_active()}
    
    def get_all_agents(self) -> Dict[int, 'SurvivalAgent']:
        """Get all agents from all teams"""
        all_agents = {}
        for team in self.teams.values():
            for agent in team.agents:
                all_agents[agent.agent_id] = agent
        return all_agents
    
    def get_alive_agents(self) -> Dict[int, 'SurvivalAgent']:
        """Get all currently alive agents"""
        alive_agents = {}
        for team in self.teams.values():
            for agent in team.get_alive_agents():
                alive_agents[agent.agent_id] = agent
        return alive_agents
    
    def reset_for_episode(self):
        """Reset all teams for a new episode"""
        for team in self.teams.values():
            if team.is_active():
                team.reset_for_episode()
    
    def update_after_episode(self, episode_result: EpisodeResult):
        """
        Update population after episode completion
        
        Args:
            episode_result: Results from the completed episode
        """
        # Update all teams
        teams_to_remove = []
        teams_to_split = []
        
        for team_id, team in self.teams.items():
            if team.is_active():
                team.update_after_episode(episode_result)
                
                # Check for elimination
                if team.status == TeamStatus.ELIMINATED:
                    teams_to_remove.append(team_id)
                
                # Check for splitting
                elif team.should_split():
                    teams_to_split.append(team_id)
        
        # Handle team eliminations
        for team_id in teams_to_remove:
            self._eliminate_team(team_id)
        
        # Handle team splits
        for team_id in teams_to_split:
            if team_id in self.teams:  # Team might have been eliminated
                self._split_team(team_id)
        
        # Check if we need to create new teams
        self._maintain_population()
        
        self.generation += 1
    
    def _eliminate_team(self, team_id: int):
        """
        Eliminate a team from the population
        
        Args:
            team_id: ID of team to eliminate
        """
        if team_id in self.teams:
            team = self.teams[team_id]
            team.status = TeamStatus.ELIMINATED
            self.eliminated_teams.append(team)
            
            # Log elimination reason
            survivors = len(team.get_alive_agents())
            if survivors < self.config.MIN_SURVIVORS_TO_CONTINUE:
                print(f"🚫 Team {team_id} eliminated: Only {survivors} survivors (need {self.config.MIN_SURVIVORS_TO_CONTINUE})")
            else:
                print(f"🚫 Team {team_id} eliminated: Poor performance over {self.config.ELIMINATION_GENERATIONS} generations")
            
            # Remove from active teams but keep for statistics
            del self.teams[team_id]
    
    def _split_team(self, team_id: int):
        """
        Split a team into two teams
        
        Args:
            team_id: ID of team to split
        """
        if team_id not in self.teams:
            return
        
        original_team = self.teams[team_id]
        
        # Create new team from split
        new_team = original_team.split_team(self._get_next_team_id())
        
        if new_team is not None:
            self.teams[new_team.team_id] = new_team
            self.total_agents_created += len(new_team.agents)
    
    def _maintain_population(self):
        """Maintain minimum population by creating new teams if needed"""
        active_teams = self.get_active_teams()
        
        # If we have too few teams, create new ones
        min_teams = max(2, self.config.INITIAL_TEAMS // 2)
        
        while len(active_teams) < min_teams:
            # Create new team by cloning a successful existing team
            if active_teams:
                # Find best performing team
                best_team = max(active_teams.values(), 
                              key=lambda t: t.performance.average_survival_rate)
                
                new_team = self._clone_team(best_team)
                self.teams[new_team.team_id] = new_team
                active_teams[new_team.team_id] = new_team
                self.total_agents_created += len(new_team.agents)
            else:
                # No active teams, create from scratch
                new_team = Team(
                    team_id=self._get_next_team_id(),
                    config=self.config
                )
                self.teams[new_team.team_id] = new_team
                active_teams[new_team.team_id] = new_team
                self.total_agents_created += len(new_team.agents)
    
    def _clone_team(self, source_team: Team) -> Team:
        """
        Create a new team by cloning an existing successful team
        
        Args:
            source_team: Team to clone from
            
        Returns:
            New cloned team
        """
        new_team = Team(
            team_id=self._get_next_team_id(),
            config=self.config,
            initial_size=0
        )
        
        # Clone agents from source team
        for agent in source_team.agents:
            cloned_agent = agent.clone(new_team._get_next_agent_id())
            cloned_agent.team_id = new_team.team_id
            
            # Add mutations to differentiate from parent
            cloned_agent.mutate_policy(
                mutation_rate=self.config.MUTATION_RATE * 1.5
            )
            
            new_team.add_agent(cloned_agent)
        
        # Set generation
        new_team.generation = source_team.generation
        
        return new_team
    
    def get_population_stats(self) -> Dict:
        """Get comprehensive population statistics"""
        active_teams = self.get_active_teams()
        all_agents = self.get_all_agents()
        alive_agents = self.get_alive_agents()
        
        # Team statistics
        team_sizes = [len(team.agents) for team in active_teams.values()]
        team_generations = [team.generation for team in active_teams.values()]
        team_survival_rates = [team.performance.average_survival_rate 
                             for team in active_teams.values()]
        
        # Agent statistics
        agent_healths = [agent.health for agent in alive_agents.values()]
        agent_ages = [agent.age for agent in alive_agents.values()]
        
        return {
            'generation': self.generation,
            'total_teams': len(active_teams),
            'eliminated_teams': len(self.eliminated_teams),
            'total_agents': len(all_agents),
            'alive_agents': len(alive_agents),
            'total_agents_created': self.total_agents_created,
            'total_teams_created': self.total_teams_created,
            
            # Team statistics
            'average_team_size': sum(team_sizes) / len(team_sizes) if team_sizes else 0,
            'min_team_size': min(team_sizes) if team_sizes else 0,
            'max_team_size': max(team_sizes) if team_sizes else 0,
            'average_team_generation': sum(team_generations) / len(team_generations) if team_generations else 0,
            'average_survival_rate': sum(team_survival_rates) / len(team_survival_rates) if team_survival_rates else 0,
            
            # Agent statistics
            'average_agent_health': sum(agent_healths) / len(agent_healths) if agent_healths else 0,
            'average_agent_age': sum(agent_ages) / len(agent_ages) if agent_ages else 0,
            
            # Diversity metrics
            'team_size_variance': self._calculate_variance(team_sizes),
            'survival_rate_variance': self._calculate_variance(team_survival_rates)
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def get_team_rankings(self) -> List[Tuple[int, Dict]]:
        """
        Get teams ranked by performance
        
        Returns:
            List of (team_id, team_summary) tuples sorted by performance
        """
        active_teams = self.get_active_teams()
        
        team_summaries = []
        for team in active_teams.values():
            summary = team.get_team_summary()
            summary['diversity_score'] = team.get_diversity_score()
            team_summaries.append((team.team_id, summary))
        
        # Sort by average survival rate (descending)
        team_summaries.sort(key=lambda x: x[1]['average_survival_rate'], reverse=True)
        
        return team_summaries
    
    def get_evolution_insights(self) -> Dict:
        """Get insights about evolutionary trends"""
        active_teams = self.get_active_teams()
        
        if not active_teams:
            return {}
        
        # Analyze team performance trends
        growing_teams = sum(1 for team in active_teams.values() 
                          if team.performance.episodes_since_growth < 3)
        declining_teams = sum(1 for team in active_teams.values() 
                            if team.performance.episodes_since_decline < 3)
        
        # Analyze generation spread
        generations = [team.generation for team in active_teams.values()]
        min_gen = min(generations) if generations else 0
        max_gen = max(generations) if generations else 0
        
        # Analyze diversity
        diversity_scores = [team.get_diversity_score() for team in active_teams.values()]
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
        
        return {
            'growing_teams': growing_teams,
            'declining_teams': declining_teams,
            'stable_teams': len(active_teams) - growing_teams - declining_teams,
            'generation_spread': max_gen - min_gen,
            'oldest_generation': max_gen,
            'youngest_generation': min_gen,
            'average_diversity': avg_diversity,
            'high_diversity_teams': sum(1 for score in diversity_scores if score > avg_diversity),
            'elimination_rate': len(self.eliminated_teams) / self.total_teams_created if self.total_teams_created > 0 else 0
        }
    
    def save_population_state(self, directory: str):
        """Save the current state of the population"""
        import json
        from pathlib import Path
        
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save population statistics
        stats = self.get_population_stats()
        insights = self.get_evolution_insights()
        rankings = self.get_team_rankings()
        
        population_data = {
            'generation': self.generation,
            'statistics': stats,
            'insights': insights,
            'team_rankings': rankings,
            'config': self.config.to_dict()
        }
        
        with open(f"{directory}/population_gen_{self.generation}.json", 'w') as f:
            json.dump(population_data, f, indent=2)
        
        # Save best policies from top teams
        top_teams = [team_id for team_id, _ in rankings[:3]]
        for team_id in top_teams:
            if team_id in self.teams:
                self.teams[team_id].save_best_policies(directory)
    
    def get_simulation_state(self) -> Dict:
        """Get current state for web interface"""
        active_teams = self.get_active_teams()
        all_agents = self.get_all_agents()
        
        return {
            'generation': self.generation,
            'total_teams': len(active_teams),
            'total_agents': len(all_agents),
            'alive_agents': len(self.get_alive_agents()),
            'teams': {
                team_id: team.get_team_summary()
                for team_id, team in active_teams.items()
            },
            'agents': {
                agent_id: {
                    'id': agent_id,
                    'team_id': agent.team_id,
                    'position': agent.position.to_tuple(),
                    'health': agent.health,
                    'alive': agent.is_alive()
                }
                for agent_id, agent in all_agents.items()
            },
            'statistics': self.get_population_stats(),
            'insights': self.get_evolution_insights()
        }
    
    def is_population_viable(self) -> bool:
        """Check if population is still viable for evolution"""
        active_teams = self.get_active_teams()
        alive_agents = self.get_alive_agents()
        
        # Need at least 2 teams and some agents to continue
        return len(active_teams) >= 2 and len(alive_agents) > 0
    
    def force_diversity(self):
        """Force diversity by mutating some teams if population becomes too homogeneous"""
        active_teams = list(self.get_active_teams().values())
        
        if len(active_teams) < 2:
            return
        
        # Calculate average diversity
        diversity_scores = [team.get_diversity_score() for team in active_teams]
        avg_diversity = sum(diversity_scores) / len(diversity_scores)
        
        # If diversity is too low, force mutations
        if avg_diversity < 0.1:  # Threshold for low diversity
            # Mutate the most homogeneous teams
            low_diversity_teams = [team for team, score in zip(active_teams, diversity_scores) 
                                 if score < avg_diversity]
            
            for team in low_diversity_teams[:len(active_teams)//2]:
                for agent in team.agents:
                    agent.mutate_policy(
                        mutation_rate=self.config.MUTATION_RATE * 3.0,
                        mutation_strength=self.config.MUTATION_STRENGTH * 2.0
                    )

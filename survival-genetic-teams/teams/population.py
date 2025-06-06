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
        # Update all teams first
        for team_id, team in self.teams.items():
            if team.is_active():
                team.update_after_episode(episode_result)
        
        # Check for diversity preservation splitting BEFORE proportional allocation
        self._check_diversity_preservation_splitting_pre_allocation(episode_result)
        
        # Apply proportional team sizing based on survival performance
        self._allocate_team_sizes_proportionally(episode_result)
        
        # Handle team splits after sizing
        teams_to_split = []
        for team_id, team in self.teams.items():
            if team.is_active() and team.should_split():
                teams_to_split.append(team_id)
        
        for team_id in teams_to_split:
            if team_id in self.teams:
                self._split_team(team_id)
        
        # Maintain minimum population
        self._maintain_population()
        
        # Final safety check - ensure we never have just 2 teams
        self._final_diversity_check()
        
        self.generation += 1
    
    def _allocate_team_sizes_proportionally(self, episode_result: EpisodeResult):
        """
        Allocate team sizes proportionally based on survival performance
        Eliminates teams with only 1 survivor, grows successful teams
        
        Args:
            episode_result: Results from the completed episode
        """
        active_teams = self.get_active_teams()
        if not active_teams:
            return
        
        # Calculate target population size (maintain roughly same total)
        current_total = sum(len(team.agents) for team in active_teams.values())
        target_population = max(self.config.INITIAL_TEAMS * self.config.STARTING_TEAM_SIZE, 
                               current_total)
        
        # Identify viable teams (>1 survivor) and eliminate others
        viable_teams = {}
        teams_to_eliminate = []
        
        for team_id, team in active_teams.items():
            survivors = episode_result.team_survivors.get(team_id, 0)
            if survivors <= 1:  # Eliminate teams with 0-1 survivors
                teams_to_eliminate.append(team_id)
                print(f"🚫 Team {team_id} eliminated: Only {survivors} survivor(s)")
            else:
                viable_teams[team_id] = {
                    'team': team,
                    'survivors': survivors,
                    'initial_size': episode_result.team_initial_sizes.get(team_id, len(team.agents))
                }
        
        # Remove eliminated teams
        for team_id in teams_to_eliminate:
            self._eliminate_team(team_id)
        
        if not viable_teams:
            return
        
        # Calculate survival ratios for viable teams
        total_survivors = sum(data['survivors'] for data in viable_teams.values())
        
        if total_survivors == 0:
            return
        
        # Allocate new team sizes proportionally
        for team_id, data in viable_teams.items():
            team = data['team']
            survivors = data['survivors']
            
            # Calculate proportional allocation
            survival_ratio = survivors / total_survivors
            new_size = max(2, int(survival_ratio * target_population))
            
            # Ensure we don't exceed maximum team size
            new_size = min(new_size, self.config.MAX_TEAM_SIZE)
            
            # Resize the team
            self._resize_team(team, new_size)
            
            print(f"📊 Team {team_id}: {len(team.agents)} → {new_size} agents "
                  f"(survival ratio: {survival_ratio:.2f})")
    
    def _resize_team(self, team: Team, target_size: int):
        """
        Resize a team to the target size by adding or removing agents
        
        Args:
            team: Team to resize
            target_size: Desired team size
        """
        current_size = len(team.agents)
        
        if target_size > current_size:
            # Grow team by cloning best performers
            agents_to_add = target_size - current_size
            best_agents = team._get_best_agents(min(3, current_size))
            
            for _ in range(agents_to_add):
                if best_agents:
                    parent = random.choice(best_agents)
                    new_agent = parent.clone(team._get_next_agent_id())
                    # Add mutation for diversity
                    new_agent.mutate_policy(
                        mutation_rate=self.config.MUTATION_RATE * 1.2
                    )
                    team.add_agent(new_agent)
                    self.total_agents_created += 1
        
        elif target_size < current_size:
            # Shrink team by removing worst performers
            agents_to_remove = current_size - target_size
            worst_agents = team._get_worst_agents(agents_to_remove)
            
            for agent in worst_agents:
                team.remove_agent(agent.agent_id)
    
    def _check_diversity_preservation_splitting_pre_allocation(self, episode_result: EpisodeResult):
        """
        Check if only two teams would survive after elimination and split the larger one for diversity
        This runs BEFORE proportional allocation to prevent getting stuck with just 2 teams
        """
        active_teams = self.get_active_teams()
        
        # Count how many teams would survive (have >1 survivor)
        viable_teams = []
        for team_id, team in active_teams.items():
            survivors = episode_result.team_survivors.get(team_id, 0)
            if survivors > 1:  # Teams with >1 survivor will survive
                viable_teams.append((team_id, team, survivors))
        
        # If only 2 teams would survive, split the larger one
        if len(viable_teams) == 2:
            # Sort by team size to find the larger team
            viable_teams.sort(key=lambda x: len(x[1].agents), reverse=True)
            larger_team_id, larger_team, larger_survivors = viable_teams[0]
            smaller_team_id, smaller_team, smaller_survivors = viable_teams[1]
            
            # Only split if the larger team has enough agents to split meaningfully
            if len(larger_team.agents) >= 4:  # Need at least 4 to split into 2 viable teams
                print(f"🔄 Pre-allocation diversity preservation: Only 2 teams would survive")
                print(f"    Team {larger_team_id}: {larger_survivors} survivors, {len(larger_team.agents)} total")
                print(f"    Team {smaller_team_id}: {smaller_survivors} survivors, {len(smaller_team.agents)} total")
                print(f"    Splitting larger team {larger_team_id} to maintain competition")
                
                # Force split the larger team
                new_team = larger_team.split_team(self._get_next_team_id())
                if new_team is not None:
                    self.teams[new_team.team_id] = new_team
                    self.total_agents_created += len(new_team.agents)
                    
                    print(f"✅ Team {larger_team_id} split into teams "
                          f"{larger_team_id} ({len(larger_team.agents)}) and "
                          f"{new_team.team_id} ({len(new_team.agents)})")
    
    def _check_diversity_preservation_splitting(self):
        """
        Check if only two teams would survive and split the larger one for diversity
        """
        active_teams = self.get_active_teams()
        
        if len(active_teams) == 2:
            # Get the two teams and their sizes
            team_list = list(active_teams.values())
            team1, team2 = team_list[0], team_list[1]
            
            # Find the larger team
            larger_team = team1 if len(team1.agents) > len(team2.agents) else team2
            
            # Only split if the larger team has enough agents to split meaningfully
            if len(larger_team.agents) >= 4:  # Need at least 4 to split into 2 viable teams
                print(f"🔄 Diversity preservation: Splitting larger team {larger_team.team_id} "
                      f"({len(larger_team.agents)} agents) to maintain competition")
                
                # Force split the larger team
                new_team = larger_team.split_team(self._get_next_team_id())
                if new_team is not None:
                    self.teams[new_team.team_id] = new_team
                    self.total_agents_created += len(new_team.agents)
                    
                    print(f"✅ Team {larger_team.team_id} split into teams "
                          f"{larger_team.team_id} ({len(larger_team.agents)}) and "
                          f"{new_team.team_id} ({len(new_team.agents)})")
    
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
        min_teams = max(3, self.config.INITIAL_TEAMS // 2)  # Ensure at least 3 teams
        
        print(f"🔍 Population maintenance: {len(active_teams)} active teams, need minimum {min_teams}")
        
        while len(active_teams) < min_teams:
            # Create new team by cloning a successful existing team
            if active_teams:
                # Find best performing team
                best_team = max(active_teams.values(), 
                              key=lambda t: t.performance.average_survival_rate)
                
                print(f"🆕 Creating new team by cloning best team {best_team.team_id}")
                new_team = self._clone_team(best_team)
                self.teams[new_team.team_id] = new_team
                active_teams[new_team.team_id] = new_team
                self.total_agents_created += len(new_team.agents)
                
                print(f"✅ Created team {new_team.team_id} with {len(new_team.agents)} agents")
            else:
                # No active teams, create from scratch
                print(f"🆕 Creating new team from scratch")
                new_team = Team(
                    team_id=self._get_next_team_id(),
                    config=self.config
                )
                self.teams[new_team.team_id] = new_team
                active_teams[new_team.team_id] = new_team
                self.total_agents_created += len(new_team.agents)
                
                print(f"✅ Created team {new_team.team_id} with {len(new_team.agents)} agents")
    
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
    
    def _final_diversity_check(self):
        """
        Final safety check to ensure we never end up with just 2 teams
        This is the last line of defense against two-team scenarios
        """
        active_teams = self.get_active_teams()
        
        print(f"🔍 Final diversity check: {len(active_teams)} active teams")
        
        if len(active_teams) == 2:
            print(f"⚠️  WARNING: Only 2 teams remain after all processing!")
            
            # Get the two teams and their sizes
            team_list = list(active_teams.values())
            team1, team2 = team_list[0], team_list[1]
            
            print(f"    Team {team1.team_id}: {len(team1.agents)} agents")
            print(f"    Team {team2.team_id}: {len(team2.agents)} agents")
            
            # Find the larger team
            larger_team = team1 if len(team1.agents) > len(team2.agents) else team2
            
            # Split the larger team if possible
            if len(larger_team.agents) >= 4:
                print(f"🔄 Final diversity preservation: Splitting team {larger_team.team_id}")
                
                new_team = larger_team.split_team(self._get_next_team_id())
                if new_team is not None:
                    self.teams[new_team.team_id] = new_team
                    self.total_agents_created += len(new_team.agents)
                    
                    print(f"✅ Emergency split: Team {larger_team.team_id} → teams "
                          f"{larger_team.team_id} ({len(larger_team.agents)}) and "
                          f"{new_team.team_id} ({len(new_team.agents)})")
            else:
                # If we can't split, create a new team by cloning
                print(f"🆕 Cannot split teams, creating new team by cloning")
                best_team = max(active_teams.values(), 
                              key=lambda t: t.performance.average_survival_rate)
                
                new_team = self._clone_team(best_team)
                self.teams[new_team.team_id] = new_team
                self.total_agents_created += len(new_team.agents)
                
                print(f"✅ Emergency clone: Created team {new_team.team_id} with {len(new_team.agents)} agents")
        
        final_count = len(self.get_active_teams())
        print(f"🏁 Final team count: {final_count}")

"""
Performance tracking and metrics for the Multi-Agent Genetic Team Survival System
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Track performance metrics for teams and overall simulation"""
    
    # Episode-level metrics
    episode_durations: List[float] = field(default_factory=list)
    episode_survivors: List[Dict[int, int]] = field(default_factory=list)  # team_id -> survivors
    episode_eliminations: List[List[int]] = field(default_factory=list)  # eliminated team_ids
    
    # Team-level metrics
    team_survival_history: Dict[int, List[float]] = field(default_factory=dict)
    team_size_history: Dict[int, List[int]] = field(default_factory=dict)
    team_generation_history: Dict[int, List[int]] = field(default_factory=dict)
    
    # Population-level metrics
    total_population_history: List[int] = field(default_factory=list)
    active_teams_history: List[int] = field(default_factory=list)
    diversity_scores: List[float] = field(default_factory=list)
    
    # Performance tracking
    simulation_start_time: Optional[float] = None
    last_episode_time: Optional[float] = None
    
    def start_simulation(self):
        """Mark the start of simulation"""
        self.simulation_start_time = time.time()
        self.last_episode_time = self.simulation_start_time
    
    def record_episode(self, episode_result, teams: Dict):
        """Record metrics from a completed episode"""
        current_time = time.time()
        
        # Episode duration
        if self.last_episode_time:
            duration = current_time - self.last_episode_time
            self.episode_durations.append(duration)
        
        self.last_episode_time = current_time
        
        # Survivors and eliminations
        self.episode_survivors.append(episode_result.team_survivors.copy())
        self.episode_eliminations.append(episode_result.team_eliminations.copy())
        
        # Team-specific metrics
        for team_id, team in teams.items():
            # Initialize team tracking if new
            if team_id not in self.team_survival_history:
                self.team_survival_history[team_id] = []
                self.team_size_history[team_id] = []
                self.team_generation_history[team_id] = []
            
            # Record survival rate
            survival_rate = episode_result.get_survival_rate(team_id)
            self.team_survival_history[team_id].append(survival_rate)
            
            # Record team size and generation
            self.team_size_history[team_id].append(len(team.agents))
            self.team_generation_history[team_id].append(team.generation)
        
        # Population metrics
        total_agents = sum(len(team.agents) for team in teams.values())
        active_teams = len([t for t in teams.values() if len(t.agents) > 0])
        
        self.total_population_history.append(total_agents)
        self.active_teams_history.append(active_teams)
        
        # Calculate diversity (simple measure based on team size variance)
        if teams:
            team_sizes = [len(team.agents) for team in teams.values()]
            if len(team_sizes) > 1:
                mean_size = sum(team_sizes) / len(team_sizes)
                variance = sum((size - mean_size) ** 2 for size in team_sizes) / len(team_sizes)
                diversity = variance / (mean_size + 1)  # Normalized diversity
            else:
                diversity = 0.0
            self.diversity_scores.append(diversity)
    
    def get_team_performance_summary(self, team_id: int) -> Dict:
        """Get performance summary for a specific team"""
        if team_id not in self.team_survival_history:
            return {}
        
        survival_rates = self.team_survival_history[team_id]
        size_history = self.team_size_history[team_id]
        generation_history = self.team_generation_history[team_id]
        
        return {
            'team_id': team_id,
            'episodes': len(survival_rates),
            'average_survival_rate': sum(survival_rates) / len(survival_rates) if survival_rates else 0,
            'best_survival_rate': max(survival_rates) if survival_rates else 0,
            'worst_survival_rate': min(survival_rates) if survival_rates else 0,
            'current_size': size_history[-1] if size_history else 0,
            'max_size': max(size_history) if size_history else 0,
            'current_generation': generation_history[-1] if generation_history else 0,
            'total_eliminations': sum(1 for survivors in self.episode_survivors 
                                    if team_id in survivors and survivors[team_id] == 0)
        }
    
    def get_simulation_summary(self) -> Dict:
        """Get overall simulation performance summary"""
        if not self.episode_durations:
            return {}
        
        total_episodes = len(self.episode_durations)
        total_time = time.time() - self.simulation_start_time if self.simulation_start_time else 0
        
        return {
            'total_episodes': total_episodes,
            'total_simulation_time': total_time,
            'average_episode_duration': sum(self.episode_durations) / len(self.episode_durations),
            'episodes_per_minute': (total_episodes / total_time) * 60 if total_time > 0 else 0,
            'current_population': self.total_population_history[-1] if self.total_population_history else 0,
            'max_population': max(self.total_population_history) if self.total_population_history else 0,
            'current_active_teams': self.active_teams_history[-1] if self.active_teams_history else 0,
            'max_active_teams': max(self.active_teams_history) if self.active_teams_history else 0,
            'average_diversity': sum(self.diversity_scores) / len(self.diversity_scores) if self.diversity_scores else 0
        }
    
    def get_recent_trends(self, window_size: int = 10) -> Dict:
        """Get trends from the most recent episodes"""
        if len(self.episode_survivors) < window_size:
            window_size = len(self.episode_survivors)
        
        if window_size == 0:
            return {}
        
        recent_survivors = self.episode_survivors[-window_size:]
        recent_population = self.total_population_history[-window_size:]
        recent_teams = self.active_teams_history[-window_size:]
        
        # Calculate trends
        population_trend = "stable"
        if len(recent_population) >= 2:
            if recent_population[-1] > recent_population[0] * 1.1:
                population_trend = "growing"
            elif recent_population[-1] < recent_population[0] * 0.9:
                population_trend = "declining"
        
        team_trend = "stable"
        if len(recent_teams) >= 2:
            if recent_teams[-1] > recent_teams[0]:
                team_trend = "diversifying"
            elif recent_teams[-1] < recent_teams[0]:
                team_trend = "consolidating"
        
        return {
            'window_size': window_size,
            'population_trend': population_trend,
            'team_count_trend': team_trend,
            'average_survivors_per_episode': sum(sum(survivors.values()) for survivors in recent_survivors) / window_size,
            'elimination_rate': sum(len(elims) for elims in self.episode_eliminations[-window_size:]) / window_size
        }
    
    def save_to_file(self, filepath: str):
        """Save metrics to JSON file"""
        data = {
            'simulation_summary': self.get_simulation_summary(),
            'team_summaries': {
                team_id: self.get_team_performance_summary(team_id)
                for team_id in self.team_survival_history.keys()
            },
            'recent_trends': self.get_recent_trends(),
            'raw_data': {
                'episode_durations': self.episode_durations,
                'episode_survivors': self.episode_survivors,
                'episode_eliminations': self.episode_eliminations,
                'team_survival_history': self.team_survival_history,
                'team_size_history': self.team_size_history,
                'team_generation_history': self.team_generation_history,
                'total_population_history': self.total_population_history,
                'active_teams_history': self.active_teams_history,
                'diversity_scores': self.diversity_scores
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        raw_data = data.get('raw_data', {})
        
        self.episode_durations = raw_data.get('episode_durations', [])
        self.episode_survivors = raw_data.get('episode_survivors', [])
        self.episode_eliminations = raw_data.get('episode_eliminations', [])
        self.team_survival_history = {
            int(k): v for k, v in raw_data.get('team_survival_history', {}).items()
        }
        self.team_size_history = {
            int(k): v for k, v in raw_data.get('team_size_history', {}).items()
        }
        self.team_generation_history = {
            int(k): v for k, v in raw_data.get('team_generation_history', {}).items()
        }
        self.total_population_history = raw_data.get('total_population_history', [])
        self.active_teams_history = raw_data.get('active_teams_history', [])
        self.diversity_scores = raw_data.get('diversity_scores', [])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for web interface"""
        return {
            'simulation_summary': self.get_simulation_summary(),
            'recent_trends': self.get_recent_trends(),
            'team_summaries': {
                team_id: self.get_team_performance_summary(team_id)
                for team_id in self.team_survival_history.keys()
            },
            'charts_data': {
                'population_history': self.total_population_history[-50:],  # Last 50 episodes
                'active_teams_history': self.active_teams_history[-50:],
                'diversity_scores': self.diversity_scores[-50:],
                'team_survival_trends': {
                    team_id: history[-20:]  # Last 20 episodes per team
                    for team_id, history in self.team_survival_history.items()
                }
            }
        }

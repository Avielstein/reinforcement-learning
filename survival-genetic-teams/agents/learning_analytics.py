"""
Learning analytics for tracking agent and team learning progress
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class PolicySnapshot:
    """Snapshot of a policy at a specific time"""
    episode: int
    timestamp: float
    weights_hash: str
    performance_score: float
    diversity_score: float
    
class LearningAnalytics:
    """
    Tracks learning progress and behavioral evolution of agents and teams
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize learning analytics
        
        Args:
            max_history: Maximum number of episodes to keep in history
        """
        self.max_history = max_history
        
        # Episode-level tracking
        self.episode_data = deque(maxlen=max_history)
        
        # Agent-level tracking
        self.agent_learning_curves = defaultdict(lambda: deque(maxlen=100))
        self.agent_policy_evolution = defaultdict(list)
        self.agent_behavioral_patterns = defaultdict(list)
        
        # Team-level tracking
        self.team_performance_history = defaultdict(lambda: deque(maxlen=100))
        self.team_coordination_scores = defaultdict(lambda: deque(maxlen=100))
        self.team_strategy_evolution = defaultdict(list)
        
        # Population-level tracking
        self.population_diversity_history = deque(maxlen=max_history)
        self.innovation_events = []
        self.extinction_events = []
        
        # Learning detection metrics
        self.learning_indicators = {
            'policy_stability': {},
            'performance_improvement': {},
            'behavioral_consistency': {},
            'strategy_emergence': {}
        }
    
    def record_episode_start(self, episode: int, teams: Dict, agents: Dict):
        """Record the start of a new episode"""
        episode_data = {
            'episode': episode,
            'timestamp': datetime.now().timestamp(),
            'start_teams': len(teams),
            'start_agents': len(agents),
            'team_sizes': {tid: len(team.agents) for tid, team in teams.items()},
            'agent_teams': {aid: agent.team_id for aid, agent in agents.items()}
        }
        
        # Initialize episode tracking
        self.current_episode_data = episode_data
    
    def record_agent_step(self, agent_id: int, observation: np.ndarray, 
                         action: np.ndarray, reward: float, 
                         policy_weights: Optional[np.ndarray] = None):
        """Record a single step for an agent"""
        step_data = {
            'observation': observation.copy() if observation is not None else None,
            'action': action.copy() if action is not None else None,
            'reward': reward,
            'timestamp': datetime.now().timestamp()
        }
        
        # Track policy evolution if weights provided
        if policy_weights is not None:
            weights_hash = self._hash_weights(policy_weights)
            policy_snapshot = PolicySnapshot(
                episode=self.current_episode_data['episode'],
                timestamp=step_data['timestamp'],
                weights_hash=weights_hash,
                performance_score=reward,
                diversity_score=self._calculate_policy_diversity(agent_id, policy_weights)
            )
            self.agent_policy_evolution[agent_id].append(policy_snapshot)
        
        # Add to agent learning curve
        self.agent_learning_curves[agent_id].append(step_data)
    
    def record_episode_end(self, episode_result: Dict, teams: Dict, agents: Dict):
        """Record the end of an episode with results"""
        # Complete episode data
        self.current_episode_data.update({
            'end_teams': len(teams),
            'end_agents': len(agents),
            'survivors_by_team': episode_result.get('team_survivors', {}),
            'eliminations': episode_result.get('team_eliminations', []),
            'total_steps': episode_result.get('total_steps', 0),
            'combat_events': episode_result.get('combat_events', 0)
        })
        
        # Calculate episode-level learning metrics
        self._calculate_episode_learning_metrics()
        
        # Store completed episode data
        self.episode_data.append(self.current_episode_data.copy())
        
        # Update team performance histories
        self._update_team_performance_tracking(teams)
        
        # Detect learning patterns
        self._detect_learning_patterns()
        
        # Track population diversity
        self._track_population_diversity(teams)
        
        # Detect innovation and extinction events
        self._detect_evolutionary_events(teams)
    
    def _calculate_episode_learning_metrics(self):
        """Calculate learning metrics for the completed episode"""
        episode = self.current_episode_data['episode']
        
        # Calculate average reward improvement for each agent
        agent_improvements = {}
        for agent_id, learning_curve in self.agent_learning_curves.items():
            if len(learning_curve) >= 10:  # Need sufficient data
                recent_rewards = [step['reward'] for step in list(learning_curve)[-10:]]
                early_rewards = [step['reward'] for step in list(learning_curve)[:10]]
                
                if early_rewards and recent_rewards:
                    improvement = np.mean(recent_rewards) - np.mean(early_rewards)
                    agent_improvements[agent_id] = improvement
        
        self.current_episode_data['agent_improvements'] = agent_improvements
        
        # Calculate policy stability for each agent
        policy_stability = {}
        for agent_id, policy_history in self.agent_policy_evolution.items():
            if len(policy_history) >= 5:
                recent_hashes = [p.weights_hash for p in policy_history[-5:]]
                stability = len(set(recent_hashes)) / len(recent_hashes)  # Lower = more stable
                policy_stability[agent_id] = 1.0 - stability
        
        self.current_episode_data['policy_stability'] = policy_stability
    
    def _update_team_performance_tracking(self, teams: Dict):
        """Update team-level performance tracking"""
        for team_id, team in teams.items():
            performance_data = {
                'episode': self.current_episode_data['episode'],
                'survival_rate': team.performance.average_survival_rate,
                'size': len(team.agents),
                'generation': team.generation,
                'diversity_score': team.get_diversity_score(),
                'coordination_score': self._calculate_team_coordination(team)
            }
            
            self.team_performance_history[team_id].append(performance_data)
            
            # Track coordination specifically
            coordination_score = performance_data['coordination_score']
            self.team_coordination_scores[team_id].append(coordination_score)
    
    def _calculate_team_coordination(self, team) -> float:
        """Calculate a coordination score for a team based on agent behaviors"""
        if len(team.agents) < 2:
            return 0.0
        
        # Simple coordination metric based on policy similarity
        # In a real implementation, this would analyze actual behavioral patterns
        coordination_score = 0.0
        
        # Get recent performance data for team agents
        team_agent_rewards = []
        for agent in team.agents:
            if agent.agent_id in self.agent_learning_curves:
                recent_steps = list(self.agent_learning_curves[agent.agent_id])[-10:]
                if recent_steps:
                    avg_reward = np.mean([step['reward'] for step in recent_steps])
                    team_agent_rewards.append(avg_reward)
        
        if len(team_agent_rewards) >= 2:
            # Lower variance in performance indicates better coordination
            reward_variance = np.var(team_agent_rewards)
            # Normalize to 0-1 scale (lower variance = higher coordination)
            coordination_score = max(0.0, 1.0 - (reward_variance / 100.0))
        
        return coordination_score
    
    def _detect_learning_patterns(self):
        """Detect various learning patterns in the data"""
        if len(self.episode_data) < 5:
            return
        
        # Detect performance improvement trends
        recent_episodes = list(self.episode_data)[-5:]
        
        for agent_id in self.agent_learning_curves.keys():
            # Check for consistent improvement
            improvements = []
            for episode_data in recent_episodes:
                if agent_id in episode_data.get('agent_improvements', {}):
                    improvements.append(episode_data['agent_improvements'][agent_id])
            
            if len(improvements) >= 3:
                trend = np.polyfit(range(len(improvements)), improvements, 1)[0]
                self.learning_indicators['performance_improvement'][agent_id] = {
                    'trend': trend,
                    'is_learning': trend > 0.01,  # Positive trend threshold
                    'confidence': min(1.0, abs(trend) * 10)
                }
            
            # Check for policy stability
            stability_scores = []
            for episode_data in recent_episodes:
                if agent_id in episode_data.get('policy_stability', {}):
                    stability_scores.append(episode_data['policy_stability'][agent_id])
            
            if stability_scores:
                avg_stability = np.mean(stability_scores)
                self.learning_indicators['policy_stability'][agent_id] = {
                    'stability': avg_stability,
                    'is_converged': avg_stability > 0.8,
                    'is_exploring': avg_stability < 0.3
                }
    
    def _track_population_diversity(self, teams: Dict):
        """Track population-level diversity metrics"""
        if not teams:
            return
        
        # Calculate team diversity scores
        team_diversities = [team.get_diversity_score() for team in teams.values()]
        
        # Calculate inter-team diversity (how different teams are from each other)
        inter_team_diversity = np.var(team_diversities) if len(team_diversities) > 1 else 0.0
        
        diversity_data = {
            'episode': self.current_episode_data['episode'],
            'avg_team_diversity': np.mean(team_diversities),
            'inter_team_diversity': inter_team_diversity,
            'num_teams': len(teams),
            'total_agents': sum(len(team.agents) for team in teams.values())
        }
        
        self.population_diversity_history.append(diversity_data)
    
    def _detect_evolutionary_events(self, teams: Dict):
        """Detect significant evolutionary events"""
        current_episode = self.current_episode_data['episode']
        
        # Detect team eliminations
        eliminations = self.current_episode_data.get('eliminations', [])
        for team_id in eliminations:
            extinction_event = {
                'episode': current_episode,
                'type': 'team_extinction',
                'team_id': team_id,
                'timestamp': datetime.now().timestamp()
            }
            self.extinction_events.append(extinction_event)
        
        # Detect innovation (significant performance jumps)
        if len(self.episode_data) >= 2:
            prev_episode = list(self.episode_data)[-1]
            
            for team_id, team in teams.items():
                if team_id in self.team_performance_history:
                    recent_performance = list(self.team_performance_history[team_id])
                    if len(recent_performance) >= 2:
                        current_perf = recent_performance[-1]['survival_rate']
                        prev_perf = recent_performance[-2]['survival_rate']
                        
                        # Significant improvement threshold
                        if current_perf - prev_perf > 0.3:
                            innovation_event = {
                                'episode': current_episode,
                                'type': 'performance_breakthrough',
                                'team_id': team_id,
                                'improvement': current_perf - prev_perf,
                                'timestamp': datetime.now().timestamp()
                            }
                            self.innovation_events.append(innovation_event)
    
    def _hash_weights(self, weights: np.ndarray) -> str:
        """Create a hash of policy weights for tracking changes"""
        # Simple hash based on rounded weights
        rounded_weights = np.round(weights, decimals=3)
        return str(hash(rounded_weights.tobytes()))
    
    def _calculate_policy_diversity(self, agent_id: int, weights: np.ndarray) -> float:
        """Calculate how diverse this policy is compared to recent policies"""
        if agent_id not in self.agent_policy_evolution:
            return 1.0
        
        recent_policies = self.agent_policy_evolution[agent_id][-10:]
        if not recent_policies:
            return 1.0
        
        # Simple diversity metric based on weight differences
        # In practice, this would be more sophisticated
        diversity_scores = []
        for policy in recent_policies:
            # This is a placeholder - in reality we'd compare actual weights
            diversity_scores.append(0.5)  # Placeholder value
        
        return np.mean(diversity_scores) if diversity_scores else 1.0
    
    def get_learning_summary(self) -> Dict:
        """Get a comprehensive summary of learning progress"""
        if not self.episode_data:
            return {'status': 'no_data'}
        
        recent_episodes = list(self.episode_data)[-10:]
        
        # Agent learning summary
        learning_agents = 0
        converged_agents = 0
        exploring_agents = 0
        
        for agent_id, indicators in self.learning_indicators['performance_improvement'].items():
            if indicators.get('is_learning', False):
                learning_agents += 1
        
        for agent_id, indicators in self.learning_indicators['policy_stability'].items():
            if indicators.get('is_converged', False):
                converged_agents += 1
            elif indicators.get('is_exploring', False):
                exploring_agents += 1
        
        # Population trends
        population_trend = 'stable'
        if len(self.population_diversity_history) >= 5:
            recent_diversity = [d['avg_team_diversity'] for d in list(self.population_diversity_history)[-5:]]
            diversity_trend = np.polyfit(range(len(recent_diversity)), recent_diversity, 1)[0]
            
            if diversity_trend > 0.01:
                population_trend = 'diversifying'
            elif diversity_trend < -0.01:
                population_trend = 'converging'
        
        return {
            'status': 'active',
            'episodes_tracked': len(self.episode_data),
            'agents_learning': learning_agents,
            'agents_converged': converged_agents,
            'agents_exploring': exploring_agents,
            'population_trend': population_trend,
            'recent_innovations': len([e for e in self.innovation_events 
                                     if e['episode'] >= recent_episodes[0]['episode']]),
            'recent_extinctions': len([e for e in self.extinction_events 
                                      if e['episode'] >= recent_episodes[0]['episode']]),
            'avg_team_coordination': np.mean([
                list(scores)[-1] if scores else 0.0 
                for scores in self.team_coordination_scores.values()
            ]) if self.team_coordination_scores else 0.0
        }
    
    def get_team_learning_analysis(self, team_id: int) -> Dict:
        """Get detailed learning analysis for a specific team"""
        if team_id not in self.team_performance_history:
            return {'status': 'no_data'}
        
        performance_history = list(self.team_performance_history[team_id])
        coordination_history = list(self.team_coordination_scores[team_id])
        
        if len(performance_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Performance trend
        survival_rates = [p['survival_rate'] for p in performance_history]
        performance_trend = np.polyfit(range(len(survival_rates)), survival_rates, 1)[0]
        
        # Coordination trend
        coordination_trend = 0.0
        if len(coordination_history) >= 2:
            coordination_trend = np.polyfit(range(len(coordination_history)), coordination_history, 1)[0]
        
        # Stability analysis
        recent_performance = survival_rates[-5:] if len(survival_rates) >= 5 else survival_rates
        performance_stability = 1.0 - (np.std(recent_performance) / (np.mean(recent_performance) + 0.001))
        
        return {
            'status': 'analyzed',
            'performance_trend': performance_trend,
            'coordination_trend': coordination_trend,
            'performance_stability': performance_stability,
            'current_survival_rate': survival_rates[-1],
            'current_coordination': coordination_history[-1] if coordination_history else 0.0,
            'episodes_tracked': len(performance_history),
            'is_improving': performance_trend > 0.01,
            'is_stable': performance_stability > 0.7
        }
    
    def export_analytics_data(self, filepath: str):
        """Export all analytics data to a JSON file"""
        export_data = {
            'episode_data': list(self.episode_data),
            'team_performance_history': {
                str(k): list(v) for k, v in self.team_performance_history.items()
            },
            'population_diversity_history': list(self.population_diversity_history),
            'innovation_events': self.innovation_events,
            'extinction_events': self.extinction_events,
            'learning_indicators': self.learning_indicators,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

"""
Experience storage and learning for survival agents
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

@dataclass
class Experience:
    """Single experience step"""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: Optional[np.ndarray]
    done: bool
    step: int

@dataclass
class EpisodeExperience:
    """Complete episode experience for an agent"""
    agent_id: int
    team_id: int
    experiences: List[Experience] = field(default_factory=list)
    total_reward: float = 0.0
    survival_time: int = 0
    final_health: float = 0.0
    
    def add_experience(self, experience: Experience):
        """Add a new experience to the episode"""
        self.experiences.append(experience)
        self.total_reward += experience.reward
        self.survival_time = experience.step
        
        # Update final health from observation (health is at index 2)
        if len(experience.observation) > 2:
            self.final_health = experience.observation[2] * 100.0  # Denormalize
    
    def get_returns(self, gamma: float = 0.99) -> List[float]:
        """Calculate discounted returns for each step"""
        returns = []
        running_return = 0.0
        
        # Calculate returns backwards
        for experience in reversed(self.experiences):
            running_return = experience.reward + gamma * running_return
            returns.append(running_return)
        
        returns.reverse()
        return returns
    
    def get_advantages(self, gamma: float = 0.99, lambda_gae: float = 0.95) -> List[float]:
        """Calculate GAE advantages"""
        if len(self.experiences) == 0:
            return []
        
        # Simple advantage calculation (reward - baseline)
        # For genetic evolution, we use survival-based rewards
        advantages = []
        baseline = self.total_reward / len(self.experiences) if self.experiences else 0
        
        for experience in self.experiences:
            advantage = experience.reward - baseline
            advantages.append(advantage)
        
        return advantages
    
    def get_policy_gradient_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data for policy gradient updates"""
        if not self.experiences:
            return np.array([]), np.array([]), np.array([])
        
        observations = np.array([exp.observation for exp in self.experiences])
        actions = np.array([exp.action for exp in self.experiences])
        advantages = np.array(self.get_advantages())
        
        return observations, actions, advantages

class ExperienceBuffer:
    """Buffer to store and manage agent experiences"""
    
    def __init__(self, max_episodes: int = 100):
        """
        Initialize experience buffer
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.max_episodes = max_episodes
        self.episodes: deque = deque(maxlen=max_episodes)
        self.current_episode: Optional[EpisodeExperience] = None
    
    def start_episode(self, agent_id: int, team_id: int):
        """Start a new episode"""
        self.current_episode = EpisodeExperience(agent_id=agent_id, team_id=team_id)
    
    def add_experience(self, observation: np.ndarray, action: np.ndarray, 
                      reward: float, next_observation: Optional[np.ndarray], 
                      done: bool, step: int):
        """Add experience to current episode"""
        if self.current_episode is None:
            raise ValueError("No episode started. Call start_episode() first.")
        
        experience = Experience(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            step=step
        )
        
        self.current_episode.add_experience(experience)
    
    def end_episode(self):
        """End current episode and store it"""
        if self.current_episode is not None:
            self.episodes.append(self.current_episode)
            self.current_episode = None
    
    def get_recent_episodes(self, n: int = 10) -> List[EpisodeExperience]:
        """Get the n most recent episodes"""
        return list(self.episodes)[-n:]
    
    def get_all_episodes(self) -> List[EpisodeExperience]:
        """Get all stored episodes"""
        return list(self.episodes)
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics from stored episodes"""
        if not self.episodes:
            return {}
        
        episodes = list(self.episodes)
        
        # Basic metrics
        total_rewards = [ep.total_reward for ep in episodes]
        survival_times = [ep.survival_time for ep in episodes]
        final_healths = [ep.final_health for ep in episodes]
        
        return {
            'total_episodes': len(episodes),
            'average_reward': np.mean(total_rewards),
            'best_reward': np.max(total_rewards),
            'worst_reward': np.min(total_rewards),
            'average_survival_time': np.mean(survival_times),
            'max_survival_time': np.max(survival_times),
            'average_final_health': np.mean(final_healths),
            'survival_rate': sum(1 for ep in episodes if ep.survival_time > 0) / len(episodes)
        }
    
    def get_learning_data(self, recent_episodes: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for policy learning from recent episodes
        
        Args:
            recent_episodes: Number of recent episodes to use
            
        Returns:
            Tuple of (observations, actions, advantages)
        """
        recent_eps = self.get_recent_episodes(recent_episodes)
        
        all_observations = []
        all_actions = []
        all_advantages = []
        
        for episode in recent_eps:
            obs, actions, advantages = episode.get_policy_gradient_data()
            if len(obs) > 0:
                all_observations.append(obs)
                all_actions.append(actions)
                all_advantages.append(advantages)
        
        if not all_observations:
            return np.array([]), np.array([]), np.array([])
        
        return (
            np.concatenate(all_observations),
            np.concatenate(all_actions),
            np.concatenate(all_advantages)
        )
    
    def clear(self):
        """Clear all stored episodes"""
        self.episodes.clear()
        self.current_episode = None

class TeamExperienceManager:
    """Manages experiences for all agents in a team"""
    
    def __init__(self, team_id: int):
        """
        Initialize team experience manager
        
        Args:
            team_id: ID of the team this manager belongs to
        """
        self.team_id = team_id
        self.agent_buffers: Dict[int, ExperienceBuffer] = {}
    
    def add_agent(self, agent_id: int):
        """Add a new agent to track"""
        if agent_id not in self.agent_buffers:
            self.agent_buffers[agent_id] = ExperienceBuffer()
    
    def remove_agent(self, agent_id: int):
        """Remove an agent from tracking"""
        if agent_id in self.agent_buffers:
            del self.agent_buffers[agent_id]
    
    def start_episode_for_all(self):
        """Start new episode for all agents"""
        for agent_id, buffer in self.agent_buffers.items():
            buffer.start_episode(agent_id, self.team_id)
    
    def end_episode_for_all(self):
        """End episode for all agents"""
        for buffer in self.agent_buffers.values():
            buffer.end_episode()
    
    def add_agent_experience(self, agent_id: int, observation: np.ndarray, 
                           action: np.ndarray, reward: float, 
                           next_observation: Optional[np.ndarray], 
                           done: bool, step: int):
        """Add experience for a specific agent"""
        if agent_id in self.agent_buffers:
            self.agent_buffers[agent_id].add_experience(
                observation, action, reward, next_observation, done, step
            )
    
    def get_team_performance_summary(self) -> Dict:
        """Get performance summary for the entire team"""
        if not self.agent_buffers:
            return {}
        
        team_metrics = {
            'team_id': self.team_id,
            'num_agents': len(self.agent_buffers),
            'agent_performances': {}
        }
        
        all_rewards = []
        all_survival_times = []
        
        for agent_id, buffer in self.agent_buffers.items():
            agent_metrics = buffer.calculate_performance_metrics()
            team_metrics['agent_performances'][agent_id] = agent_metrics
            
            if agent_metrics:
                all_rewards.extend([ep.total_reward for ep in buffer.get_all_episodes()])
                all_survival_times.extend([ep.survival_time for ep in buffer.get_all_episodes()])
        
        if all_rewards:
            team_metrics['team_average_reward'] = np.mean(all_rewards)
            team_metrics['team_average_survival'] = np.mean(all_survival_times)
            team_metrics['team_best_performance'] = np.max(all_rewards)
        
        return team_metrics
    
    def get_team_learning_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get learning data for all agents in the team"""
        learning_data = {}
        
        for agent_id, buffer in self.agent_buffers.items():
            obs, actions, advantages = buffer.get_learning_data()
            if len(obs) > 0:
                learning_data[agent_id] = (obs, actions, advantages)
        
        return learning_data
    
    def calculate_team_diversity(self) -> float:
        """Calculate behavioral diversity within the team"""
        if len(self.agent_buffers) < 2:
            return 0.0
        
        # Simple diversity measure based on action variance
        all_actions = []
        
        for buffer in self.agent_buffers.values():
            recent_episodes = buffer.get_recent_episodes(3)
            for episode in recent_episodes:
                for experience in episode.experiences:
                    all_actions.append(experience.action)
        
        if len(all_actions) < 2:
            return 0.0
        
        actions_array = np.array(all_actions)
        # Calculate variance across all action dimensions
        variances = np.var(actions_array, axis=0)
        return np.mean(variances)

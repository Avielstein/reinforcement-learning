"""
Episode runner for the Multi-Agent Genetic Team Survival System
"""

import time
from typing import Dict, List, Optional, Callable
import threading
import queue

from core.config import Config
from core.types import AgentAction, ActionType, SimulationState
from core.metrics import PerformanceMetrics
from teams.population import Population
from environment.survival_env import SurvivalEnvironment

class EpisodeRunner:
    """
    Manages the execution of episodes in the survival simulation
    """
    
    def __init__(self, config: Config):
        """
        Initialize episode runner
        
        Args:
            config: Global configuration
        """
        self.config = config
        self.population = Population(config)
        self.environment = SurvivalEnvironment(config)
        self.metrics = PerformanceMetrics()
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.current_episode = 0
        self.current_step = 0
        
        # Threading for non-blocking execution
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks for real-time updates
        self.step_callbacks: List[Callable] = []
        self.episode_callbacks: List[Callable] = []
        
        # Performance tracking
        self.episode_start_time = None
        self.total_simulation_time = 0.0
    
    def add_step_callback(self, callback: Callable):
        """Add callback to be called after each step"""
        self.step_callbacks.append(callback)
    
    def add_episode_callback(self, callback: Callable):
        """Add callback to be called after each episode"""
        self.episode_callbacks.append(callback)
    
    def start_simulation(self, max_episodes: Optional[int] = None, 
                        background: bool = True) -> bool:
        """
        Start the simulation
        
        Args:
            max_episodes: Maximum number of episodes to run (None for infinite)
            background: Whether to run in background thread
            
        Returns:
            True if simulation started successfully
        """
        if self.is_running:
            return False
        
        self.is_running = True
        self.is_paused = False
        self.stop_event.clear()
        self.metrics.start_simulation()
        
        if background:
            self.simulation_thread = threading.Thread(
                target=self._run_simulation_loop,
                args=(max_episodes,),
                daemon=True
            )
            self.simulation_thread.start()
        else:
            self._run_simulation_loop(max_episodes)
        
        return True
    
    def pause_simulation(self):
        """Pause the simulation"""
        self.is_paused = True
    
    def resume_simulation(self):
        """Resume the simulation"""
        self.is_paused = False
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.stop_event.set()
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5.0)
    
    def step_simulation(self) -> bool:
        """
        Execute a single step of the simulation
        
        Returns:
            True if step was executed successfully
        """
        if not self.is_running or self.is_paused:
            return False
        
        # Check if we need to start a new episode
        if self.current_step == 0 or self.environment.is_episode_complete():
            if not self._start_new_episode():
                return False
        
        # Get actions from all alive agents
        actions = self._get_agent_actions()
        
        # Execute environment step
        observations, rewards, dones, info = self.environment.step(actions)
        
        # Record experiences for agents
        self._record_agent_experiences(observations, actions, rewards, dones)
        
        # Update step counter
        self.current_step += 1
        
        # Call step callbacks
        for callback in self.step_callbacks:
            try:
                callback(self.get_simulation_state())
            except Exception as e:
                print(f"Error in step callback: {e}")
        
        # Check if episode is complete
        if self.environment.is_episode_complete():
            self._end_current_episode()
        
        return True
    
    def _run_simulation_loop(self, max_episodes: Optional[int]):
        """Main simulation loop"""
        episode_count = 0
        
        while self.is_running and not self.stop_event.is_set():
            if max_episodes and episode_count >= max_episodes:
                break
            
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            # Execute simulation step
            if not self.step_simulation():
                break
            
            # Check if episode completed
            if self.current_step == 0:  # New episode started
                episode_count += 1
            
            # Configurable delay for simulation speed control
            delay = getattr(self.config, 'SIMULATION_SPEED', 0.001)
            time.sleep(delay)
        
        self.is_running = False
    
    def _start_new_episode(self) -> bool:
        """Start a new episode"""
        # Check if population is still viable
        if not self.population.is_population_viable():
            print("Population no longer viable. Ending simulation.")
            return False
        
        self.current_episode += 1
        self.current_step = 0
        self.episode_start_time = time.time()
        
        # Reset population for new episode
        active_teams = self.population.get_active_teams()
        self.population.reset_for_episode()
        
        # Reset environment
        self.environment.reset(active_teams)
        
        print(f"🧬 Starting Episode {self.current_episode}")
        print(f"   Teams: {len(active_teams)}")
        print(f"   Total Agents: {len(self.population.get_all_agents())}")
        
        return True
    
    def _end_current_episode(self):
        """End the current episode and update population"""
        # Get episode results
        episode_result = self.environment.get_episode_result()
        
        # Update population with results
        self.population.update_after_episode(episode_result)
        
        # Record metrics
        self.metrics.record_episode(episode_result, self.population.get_active_teams())
        
        # Calculate episode duration
        if self.episode_start_time:
            episode_duration = time.time() - self.episode_start_time
            self.total_simulation_time += episode_duration
        
        # Print episode summary
        self._print_episode_summary(episode_result)
        
        # Call episode callbacks
        for callback in self.episode_callbacks:
            try:
                callback(episode_result, self.population.get_simulation_state())
            except Exception as e:
                print(f"Error in episode callback: {e}")
        
        # Reset for next episode
        self.current_step = 0
    
    def _get_agent_actions(self) -> Dict[int, AgentAction]:
        """Get actions from all alive agents"""
        actions = {}
        
        for agent_id, agent in self.population.get_alive_agents().items():
            # Get agent's observation
            nearby_agents = self.environment._get_nearby_agents(agent)
            observation = agent.get_observation(nearby_agents)
            
            # Get agent's action
            action = agent.select_action(observation, add_exploration=True)
            actions[agent_id] = action
        
        return actions
    
    def _record_agent_experiences(self, observations: Dict, actions: Dict, 
                                 rewards: Dict, dones: Dict):
        """Record experiences for agent learning"""
        for agent_id, agent in self.population.get_all_agents().items():
            if agent_id in observations and agent_id in actions:
                observation = observations[agent_id]
                action = actions[agent_id]
                reward = rewards.get(agent_id, 0.0)
                done = dones.get(agent_id, False)
                
                # Get next observation (None if done)
                next_observation = None
                if not done and agent.is_alive():
                    nearby_agents = self.environment._get_nearby_agents(agent)
                    next_observation = agent.get_observation(nearby_agents)
                
                # Record experience
                agent.record_experience(observation, action, reward, next_observation, done)
    
    def _print_episode_summary(self, episode_result):
        """Print summary of completed episode"""
        print(f"📊 Episode {self.current_episode} Complete")
        print(f"   Duration: {episode_result.total_steps} steps")
        
        # Team survival summary
        for team_id, survivors in episode_result.team_survivors.items():
            initial = episode_result.team_initial_sizes[team_id]
            survival_rate = survivors / initial if initial > 0 else 0
            print(f"   Team {team_id}: {survivors}/{initial} survived ({survival_rate:.1%})")
        
        # Eliminated teams
        if episode_result.team_eliminations:
            print(f"   💀 Eliminated: Teams {episode_result.team_eliminations}")
        
        # Combat statistics
        combat_stats = self.environment.get_combat_statistics()
        if combat_stats:
            print(f"   ⚔️  Combat: {combat_stats['total_combat_events']} events, "
                  f"{combat_stats['total_kills']} kills")
        
        print()
    
    def get_simulation_state(self) -> SimulationState:
        """Get current simulation state"""
        active_teams = self.population.get_active_teams()
        all_agents = self.population.get_all_agents()
        team_stats = {}
        
        # Create team stats
        for team_id, team in active_teams.items():
            team_stats[team_id] = team.get_team_summary()
        
        return SimulationState(
            episode=self.current_episode,
            step=self.current_step,
            teams=active_teams,
            agents=all_agents,
            team_stats=team_stats,
            is_running=self.is_running and not self.is_paused,
            total_agents_alive=len(self.population.get_alive_agents())
        )
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        simulation_summary = self.metrics.get_simulation_summary()
        population_stats = self.population.get_population_stats()
        evolution_insights = self.population.get_evolution_insights()
        
        return {
            'simulation': simulation_summary,
            'population': population_stats,
            'evolution': evolution_insights,
            'current_episode': self.current_episode,
            'total_simulation_time': self.total_simulation_time,
            'is_running': self.is_running,
            'is_paused': self.is_paused
        }
    
    def save_simulation_state(self, directory: str):
        """Save current simulation state to files"""
        from pathlib import Path
        import json
        
        # Create directory
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save performance metrics
        self.metrics.save_to_file(f"{directory}/metrics.json")
        
        # Save population state
        self.population.save_population_state(directory)
        
        # Save simulation summary
        summary = self.get_performance_summary()
        with open(f"{directory}/simulation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Simulation state saved to {directory}")
    
    def force_evolution(self):
        """Force evolutionary pressure on the population"""
        self.population.force_diversity()
        print("🧬 Forced evolutionary pressure applied")
    
    def adjust_config(self, new_params: Dict):
        """Adjust configuration parameters during simulation"""
        self.config.update_from_dict(new_params)
        print(f"⚙️  Configuration updated: {new_params}")
    
    def get_real_time_stats(self) -> Dict:
        """Get real-time statistics for web interface"""
        env_state = self.environment.get_environment_state()
        pop_state = self.population.get_simulation_state()
        
        return {
            'episode': self.current_episode,
            'step': self.current_step,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'environment': env_state,
            'population': pop_state,
            'performance': self.get_performance_summary()
        }
    
    def run_single_episode(self) -> Dict:
        """Run a single episode and return results (for testing)"""
        if not self._start_new_episode():
            return {}
        
        while not self.environment.is_episode_complete():
            actions = self._get_agent_actions()
            observations, rewards, dones, info = self.environment.step(actions)
            self._record_agent_experiences(observations, actions, rewards, dones)
            self.current_step += 1
        
        episode_result = self.environment.get_episode_result()
        self.population.update_after_episode(episode_result)
        
        return {
            'episode_result': episode_result,
            'combat_stats': self.environment.get_combat_statistics(),
            'population_stats': self.population.get_population_stats()
        }

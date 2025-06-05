"""
Combat environment for genetic radar evolution
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from ..core.config import Config

class CombatEnvironment:
    """Environment for running tactical combat simulations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.map_size = config.map_size
        self.dt = config.dt
        self.max_simulation_time = config.max_simulation_time
        
        # Simulation state
        self.current_time = 0.0
        self.projectiles = []
        self.explosions = []
        
    def run_battle(self, agents: List) -> Dict[str, Any]:
        """Run a complete battle simulation"""
        self.current_time = 0.0
        self.projectiles = []
        self.explosions = []
        
        # Reset agent positions and states
        self._reset_agents(agents)
        
        # Main simulation loop
        while self.current_time < self.max_simulation_time:
            self._update_simulation_step(agents)
            self.current_time += self.dt
            
            # Check if battle is over
            active_species = self._count_active_species(agents)
            if len(active_species) <= 1:
                break
        
        # Calculate results
        return self._calculate_battle_results(agents)
    
    def _reset_agents(self, agents: List) -> None:
        """Reset agents to starting positions and states"""
        species_positions = {}
        species_counts = {}
        
        # Count agents per species
        for agent in agents:
            if agent.species_id not in species_counts:
                species_counts[agent.species_id] = 0
            species_counts[agent.species_id] += 1
        
        # Assign starting positions
        species_list = list(species_counts.keys())
        for i, agent in enumerate(agents):
            species_idx = species_list.index(agent.species_id)
            agent_idx = species_positions.get(agent.species_id, 0)
            species_positions[agent.species_id] = agent_idx + 1
            
            # Position agents around the map edges
            angle = (species_idx / len(species_list)) * 2 * np.pi
            base_x = self.map_size * 0.5 + np.cos(angle) * self.map_size * 0.3
            base_y = self.map_size * 0.5 + np.sin(angle) * self.map_size * 0.3
            
            # Spread agents within species
            offset_angle = (agent_idx / species_counts[agent.species_id]) * np.pi / 2
            offset_distance = 50
            
            agent.x = base_x + np.cos(offset_angle) * offset_distance
            agent.y = base_y + np.sin(offset_angle) * offset_distance
            
            # Keep within bounds
            agent.x = np.clip(agent.x, 50, self.map_size - 50)
            agent.y = np.clip(agent.y, 50, self.map_size - 50)
            
            # Reset agent state
            agent.health = agent.max_health
            agent.energy = agent.max_energy
            agent.reload_time_remaining = 0.0
            agent.kills = 0
            agent.damage_dealt = 0.0
            agent.damage_received = 0.0
            agent.survival_time = 0.0
            agent.shots_fired = 0
            agent.shots_hit = 0
            agent.action_history = []
            agent.position_history = []
    
    def _update_simulation_step(self, agents: List) -> None:
        """Update one simulation step"""
        # Update all agents
        for agent in agents:
            if agent.health > 0:
                action_result = agent.update(self.dt, agents, self.map_size)
                
                # Handle firing
                if action_result and 'fire_at' in action_result:
                    target = action_result['fire_at']
                    projectile_data = agent.fire_projectile(target)
                    if projectile_data:
                        self.projectiles.append(projectile_data)
        
        # Update projectiles
        self._update_projectiles(agents)
        
        # Update explosions
        self._update_explosions()
    
    def _update_projectiles(self, agents: List) -> None:
        """Update projectile positions and check for hits"""
        remaining_projectiles = []
        
        for proj in self.projectiles:
            # Move projectile
            proj['x'] += np.cos(proj['angle']) * proj['speed'] * self.dt
            proj['y'] += np.sin(proj['angle']) * proj['speed'] * self.dt
            
            # Check bounds
            if (proj['x'] < 0 or proj['x'] > self.map_size or
                proj['y'] < 0 or proj['y'] > self.map_size):
                continue
            
            # Check for hits
            hit = False
            for agent in agents:
                if (agent.health > 0 and 
                    agent.species_id != proj['species_id']):
                    
                    distance = np.sqrt((agent.x - proj['x'])**2 + (agent.y - proj['y'])**2)
                    if distance < 10:  # Hit radius
                        # Hit!
                        killed = agent.take_damage(proj['damage'], proj['shooter_id'])
                        
                        # Update shooter stats
                        shooter = next((a for a in agents if a.id == proj['shooter_id']), None)
                        if shooter:
                            shooter.shots_hit += 1
                            if killed:
                                shooter.kills += 1
                        
                        # Create explosion
                        self.explosions.append({
                            'x': proj['x'],
                            'y': proj['y'],
                            'size': 15,
                            'lifetime': 0.5
                        })
                        
                        hit = True
                        break
            
            if not hit:
                remaining_projectiles.append(proj)
        
        self.projectiles = remaining_projectiles
    
    def _update_explosions(self) -> None:
        """Update explosion effects"""
        remaining_explosions = []
        for exp in self.explosions:
            exp['lifetime'] -= self.dt
            if exp['lifetime'] > 0:
                remaining_explosions.append(exp)
        self.explosions = remaining_explosions
    
    def _count_active_species(self, agents: List) -> Dict[str, int]:
        """Count living agents per species"""
        active = {}
        for agent in agents:
            if agent.health > 0:
                if agent.species_id not in active:
                    active[agent.species_id] = 0
                active[agent.species_id] += 1
        return active
    
    def _calculate_battle_results(self, agents: List) -> Dict[str, Any]:
        """Calculate battle results"""
        results = {
            'duration': self.current_time,
            'agent_results': [],
            'species_summary': {}
        }
        
        # Individual agent results
        for agent in agents:
            agent_result = {
                'agent_id': agent.id,
                'species_id': agent.species_id,
                'survived': agent.health > 0,
                'kills': agent.kills,
                'damage_dealt': agent.damage_dealt,
                'damage_received': agent.damage_received,
                'survival_time': agent.survival_time,
                'shots_fired': agent.shots_fired,
                'shots_hit': agent.shots_hit,
                'accuracy': agent.shots_hit / max(1, agent.shots_fired)
            }
            results['agent_results'].append(agent_result)
        
        # Species summary
        species_stats = {}
        for agent in agents:
            if agent.species_id not in species_stats:
                species_stats[agent.species_id] = {
                    'total_agents': 0,
                    'survivors': 0,
                    'total_kills': 0,
                    'total_damage': 0.0
                }
            
            stats = species_stats[agent.species_id]
            stats['total_agents'] += 1
            if agent.health > 0:
                stats['survivors'] += 1
            stats['total_kills'] += agent.kills
            stats['total_damage'] += agent.damage_dealt
        
        results['species_summary'] = species_stats
        
        return results

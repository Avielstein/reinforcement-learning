"""
Simple demo of genetic radar evolution
"""

import numpy as np
import torch
import random
import time
from pathlib import Path

# Add current directory to path
import sys
sys.path.append('.')

# Import modules with absolute paths
from core.config import Config
from agents.policy_network import PolicyNetwork, SpecializedPolicyNetwork
from agents.neural_agent import NeuralAgent
from agents.behavior_analyzer import BehaviorAnalyzer, create_behavior_report

class SimpleCombatEnvironment:
    """Simplified combat environment for demo"""
    
    def __init__(self, map_size=1000.0):
        self.map_size = map_size
        self.dt = 0.1
        self.projectiles = []
    
    def run_battle(self, agents, max_time=60.0):
        """Run a simplified battle"""
        current_time = 0.0
        
        # Reset agents
        self._reset_agents(agents)
        
        # Simple battle loop
        while current_time < max_time:
            # Update agents
            for agent in agents:
                if agent.health > 0:
                    agent.update(self.dt, agents, self.map_size)
            
            current_time += self.dt
            
            # Check if battle is over
            active_species = self._count_active_species(agents)
            if len(active_species) <= 1:
                break
        
        return self._calculate_results(agents)
    
    def _reset_agents(self, agents):
        """Reset agent positions"""
        species_positions = {}
        for i, agent in enumerate(agents):
            if agent.species_id not in species_positions:
                species_positions[agent.species_id] = 0
            
            # Position agents in different corners
            if agent.species_id == 'alpha':
                agent.x = 100 + species_positions[agent.species_id] * 50
                agent.y = 100
            elif agent.species_id == 'beta':
                agent.x = self.map_size - 100 - species_positions[agent.species_id] * 50
                agent.y = self.map_size - 100
            else:
                agent.x = 100
                agent.y = self.map_size - 100 - species_positions[agent.species_id] * 50
            
            species_positions[agent.species_id] += 1
            
            # Reset stats
            agent.health = agent.max_health
            agent.kills = 0
            agent.damage_dealt = 0.0
            agent.survival_time = 0.0
    
    def _count_active_species(self, agents):
        """Count active species"""
        active = {}
        for agent in agents:
            if agent.health > 0:
                active[agent.species_id] = active.get(agent.species_id, 0) + 1
        return active
    
    def _calculate_results(self, agents):
        """Calculate battle results"""
        results = {'agent_results': []}
        
        for agent in agents:
            results['agent_results'].append({
                'agent_id': agent.id,
                'species_id': agent.species_id,
                'survived': agent.health > 0,
                'kills': agent.kills,
                'damage_dealt': agent.damage_dealt,
                'survival_time': agent.survival_time
            })
        
        return results

def run_simple_demo():
    """Run a simple demonstration"""
    print("🧬 Simple Genetic Radar Evolution Demo")
    print("=" * 50)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    env = SimpleCombatEnvironment()
    
    # Create species
    species_configs = [
        {'id': 'alpha', 'name': 'Alpha Hunters', 'strategy': 'aggressive', 'count': 3},
        {'id': 'beta', 'name': 'Beta Defenders', 'strategy': 'defensive', 'count': 3},
        {'id': 'gamma', 'name': 'Gamma Scouts', 'strategy': 'scout', 'count': 3}
    ]
    
    # Create agents
    all_agents = []
    for species_config in species_configs:
        print(f"Creating {species_config['name']} with {species_config['count']} agents")
        
        for i in range(species_config['count']):
            # Create specialized neural network
            policy = SpecializedPolicyNetwork(strategy=species_config['strategy'])
            
            agent = NeuralAgent(
                agent_id=f"{species_config['id']}_{i}",
                species_id=species_config['id'],
                policy_network=policy
            )
            
            all_agents.append(agent)
    
    print(f"\n🎯 Running battle with {len(all_agents)} agents...")
    
    # Run battle
    start_time = time.time()
    battle_result = env.run_battle(all_agents)
    battle_time = time.time() - start_time
    
    print(f"⚔️  Battle completed in {battle_time:.2f} seconds")
    
    # Analyze results
    species_stats = {}
    for result in battle_result['agent_results']:
        species_id = result['species_id']
        if species_id not in species_stats:
            species_stats[species_id] = {
                'survivors': 0,
                'total_kills': 0,
                'total_damage': 0.0,
                'avg_survival': 0.0,
                'agents': []
            }
        
        stats = species_stats[species_id]
        if result['survived']:
            stats['survivors'] += 1
        stats['total_kills'] += result['kills']
        stats['total_damage'] += result['damage_dealt']
        stats['avg_survival'] += result['survival_time']
        stats['agents'].append(result)
    
    # Calculate averages
    for species_id, stats in species_stats.items():
        agent_count = len(stats['agents'])
        stats['avg_survival'] /= max(agent_count, 1)
        stats['survival_rate'] = stats['survivors'] / max(agent_count, 1)
    
    # Print results
    print("\n📊 Battle Results:")
    print("-" * 30)
    
    for species_config in species_configs:
        species_id = species_config['id']
        name = species_config['name']
        stats = species_stats.get(species_id, {})
        
        print(f"\n{name} ({species_id}):")
        print(f"  Survivors: {stats.get('survivors', 0)}/{species_config['count']}")
        print(f"  Survival Rate: {stats.get('survival_rate', 0):.1%}")
        print(f"  Total Kills: {stats.get('total_kills', 0)}")
        print(f"  Total Damage: {stats.get('total_damage', 0):.1f}")
        print(f"  Avg Survival Time: {stats.get('avg_survival', 0):.1f}s")
    
    # Analyze behaviors
    print("\n🧠 Behavioral Analysis:")
    print("-" * 30)
    
    analyzer = BehaviorAnalyzer()
    for agent in all_agents:
        if agent.action_history:  # Only analyze agents with action history
            agent_data = {
                'behavioral_metrics': agent.get_behavioral_metrics(),
                'action_history': agent.action_history,
                'position_history': agent.position_history
            }
            
            behavior_analysis = analyzer.analyze_agent_behavior(agent_data)
            
            print(f"\nAgent {agent.id}:")
            print(f"  Strategy: {behavior_analysis['primary_strategy']}")
            print(f"  Emergent Patterns: {', '.join(behavior_analysis['emergent_patterns'])}")
            print(f"  Adaptation Score: {behavior_analysis['adaptation_score']:.3f}")
            print(f"  Uniqueness Score: {behavior_analysis['uniqueness_score']:.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("\nThis demonstrates:")
    print("• Neural network policies making tactical decisions")
    print("• Different species with specialized starting strategies")
    print("• Emergent behaviors from neural network evolution")
    print("• Automatic behavior analysis and classification")
    
    print(f"\n🚀 To run full evolution experiments:")
    print("   cd genetic-radar-evolution")
    print("   python simple_demo.py")

if __name__ == "__main__":
    run_simple_demo()

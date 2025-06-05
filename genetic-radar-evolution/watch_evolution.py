"""
Watch neural networks evolve and learn in real-time
"""

import numpy as np
import torch
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add current directory to path
import sys
sys.path.append('.')

from core.config import Config
from agents.policy_network import PolicyNetwork, SpecializedPolicyNetwork
from agents.neural_agent import NeuralAgent
from agents.behavior_analyzer import BehaviorAnalyzer

class EvolutionWatcher:
    """Watch evolution happen in real-time"""
    
    def __init__(self):
        self.map_size = 800.0
        self.dt = 0.1
        self.generation = 0
        self.species_populations = {}
        self.fitness_history = {'alpha': [], 'beta': [], 'gamma': []}
        self.behavior_history = {'alpha': [], 'beta': [], 'gamma': []}
        
        # Set random seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Initialize species
        self._initialize_species()
        
        print("🧬 Evolution Watcher Initialized")
        print("=" * 50)
        print("Watch neural networks evolve tactical behaviors!")
        print("Each generation, networks mutate and the best survive.")
        print("Look for emergent strategies to develop over time.")
        print("=" * 50)
    
    def _initialize_species(self):
        """Initialize starting species"""
        species_configs = [
            {'id': 'alpha', 'name': 'Alpha Hunters', 'strategy': 'aggressive', 'count': 4},
            {'id': 'beta', 'name': 'Beta Defenders', 'strategy': 'defensive', 'count': 4},
            {'id': 'gamma', 'name': 'Gamma Scouts', 'strategy': 'scout', 'count': 4}
        ]
        
        for species_config in species_configs:
            agents = []
            for i in range(species_config['count']):
                policy = SpecializedPolicyNetwork(strategy=species_config['strategy'])
                agent = NeuralAgent(
                    agent_id=f"{species_config['id']}_{i}",
                    species_id=species_config['id'],
                    policy_network=policy
                )
                agents.append(agent)
            
            self.species_populations[species_config['id']] = {
                'agents': agents,
                'config': species_config,
                'fitness_history': [],
                'best_fitness': 0.0
            }
    
    def run_battle(self, agents, max_time=30.0):
        """Run a battle and return fitness scores"""
        current_time = 0.0
        
        # Reset agents
        self._reset_agents(agents)
        
        # Battle loop
        while current_time < max_time:
            # Update all agents
            for agent in agents:
                if agent.health > 0:
                    agent.update(self.dt, agents, self.map_size)
            
            current_time += self.dt
            
            # Check if battle is over (only one species left)
            active_species = set()
            for agent in agents:
                if agent.health > 0:
                    active_species.add(agent.species_id)
            
            if len(active_species) <= 1:
                break
        
        # Calculate fitness for each species
        species_fitness = {}
        for species_id in self.species_populations.keys():
            species_agents = [a for a in agents if a.species_id == species_id]
            
            total_fitness = 0.0
            for agent in species_agents:
                # Fitness = survival + kills + damage + time alive
                fitness = (
                    (100 if agent.health > 0 else 0) +  # Survival bonus
                    agent.kills * 50 +                   # Kill bonus
                    agent.damage_dealt * 0.5 +           # Damage bonus
                    agent.survival_time * 0.1            # Time bonus
                )
                total_fitness += fitness
            
            species_fitness[species_id] = total_fitness / len(species_agents)
        
        return species_fitness
    
    def _reset_agents(self, agents):
        """Reset agents for battle"""
        species_positions = {}
        
        for agent in agents:
            if agent.species_id not in species_positions:
                species_positions[agent.species_id] = 0
            
            # Position species in different areas
            if agent.species_id == 'alpha':
                agent.x = 100 + species_positions[agent.species_id] * 60
                agent.y = 100
            elif agent.species_id == 'beta':
                agent.x = self.map_size - 100 - species_positions[agent.species_id] * 60
                agent.y = self.map_size - 100
            else:  # gamma
                agent.x = 100
                agent.y = self.map_size - 100 - species_positions[agent.species_id] * 60
            
            species_positions[agent.species_id] += 1
            
            # Reset stats
            agent.health = agent.max_health
            agent.energy = agent.max_energy
            agent.kills = 0
            agent.damage_dealt = 0.0
            agent.damage_received = 0.0
            agent.survival_time = 0.0
            agent.shots_fired = 0
            agent.shots_hit = 0
            agent.action_history = []
            agent.position_history = []
    
    def evolve_species(self, fitness_scores):
        """Evolve species based on fitness"""
        for species_id, species_data in self.species_populations.items():
            fitness = fitness_scores[species_id]
            species_data['fitness_history'].append(fitness)
            
            # Determine evolution type
            if len(species_data['fitness_history']) >= 3:
                recent_avg = np.mean(species_data['fitness_history'][-3:])
                if fitness < recent_avg * 0.8:
                    evolution_type = "🧬 MAJOR EVOLUTION"
                    self._major_evolution(species_id)
                elif fitness > recent_avg * 1.2:
                    evolution_type = "🔧 minor refinement"
                    self._minor_evolution(species_id)
                else:
                    evolution_type = "🎲 random mutations"
                    self._random_mutations(species_id)
            else:
                evolution_type = "🎲 random mutations"
                self._random_mutations(species_id)
            
            # Track best fitness
            if fitness > species_data['best_fitness']:
                species_data['best_fitness'] = fitness
            
            print(f"   {species_data['config']['name']}: {fitness:.1f} ({evolution_type})")
    
    def _major_evolution(self, species_id):
        """Major evolution - keep best, evolve rest"""
        species_data = self.species_populations[species_id]
        agents = species_data['agents']
        
        # Sort by individual performance
        agents.sort(key=lambda a: a.kills * 50 + a.damage_dealt + (100 if a.health > 0 else 0), reverse=True)
        
        # Keep top 25% (1 agent), evolve the rest
        elite_count = max(1, len(agents) // 4)
        elite_agents = agents[:elite_count]
        
        # Generate new agents through crossover and mutation
        new_agents = elite_agents.copy()
        
        while len(new_agents) < len(agents):
            parent1 = random.choice(elite_agents)
            parent2 = random.choice(elite_agents)
            
            # Create offspring
            offspring_policy = parent1.policy.crossover(parent2.policy, 0.7)
            offspring_policy = offspring_policy.mutate(0.3, 0.4)
            
            new_agent = NeuralAgent(
                agent_id=f"{species_id}_{len(new_agents)}",
                species_id=species_id,
                policy_network=offspring_policy
            )
            new_agents.append(new_agent)
        
        species_data['agents'] = new_agents
    
    def _minor_evolution(self, species_id):
        """Minor evolution - small mutations"""
        species_data = self.species_populations[species_id]
        for agent in species_data['agents']:
            agent.policy = agent.policy.mutate(0.1, 0.1)
    
    def _random_mutations(self, species_id):
        """Random mutations"""
        species_data = self.species_populations[species_id]
        agents_to_mutate = random.sample(species_data['agents'], len(species_data['agents']) // 2)
        for agent in agents_to_mutate:
            agent.policy = agent.policy.mutate(0.15, 0.2)
    
    def analyze_behaviors(self):
        """Analyze current behaviors"""
        analyzer = BehaviorAnalyzer()
        species_behaviors = {}
        
        for species_id, species_data in self.species_populations.items():
            behaviors = []
            strategies = []
            
            for agent in species_data['agents']:
                if agent.action_history:
                    agent_data = {
                        'behavioral_metrics': agent.get_behavioral_metrics(),
                        'action_history': agent.action_history,
                        'position_history': agent.position_history
                    }
                    
                    analysis = analyzer.analyze_agent_behavior(agent_data)
                    strategies.append(analysis['primary_strategy'])
                    behaviors.extend(analysis['emergent_patterns'])
            
            # Count strategy frequencies
            strategy_counts = {}
            for strategy in strategies:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            dominant_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else 'unknown'
            unique_behaviors = list(set(behaviors))
            
            species_behaviors[species_id] = {
                'dominant_strategy': dominant_strategy,
                'behaviors': unique_behaviors,
                'strategy_diversity': len(strategy_counts)
            }
        
        return species_behaviors
    
    def run_evolution(self, generations=20):
        """Run evolution and watch it happen"""
        print(f"\n🚀 Starting {generations} generations of evolution...")
        print("Watch for:")
        print("• Fitness scores improving over time")
        print("• New behaviors emerging")
        print("• Species adapting to each other")
        print("• Arms races developing")
        print("\n" + "=" * 60)
        
        for gen in range(generations):
            self.generation = gen
            print(f"\n🧬 Generation {gen + 1}/{generations}")
            print("-" * 40)
            
            # Collect all agents
            all_agents = []
            for species_data in self.species_populations.values():
                all_agents.extend(species_data['agents'])
            
            # Run battle
            print("⚔️  Running battle...")
            fitness_scores = self.run_battle(all_agents)
            
            # Store fitness history
            for species_id, fitness in fitness_scores.items():
                self.fitness_history[species_id].append(fitness)
            
            # Evolve based on performance
            print("🧬 Evolution results:")
            self.evolve_species(fitness_scores)
            
            # Analyze behaviors every few generations
            if gen % 3 == 0:
                print("\n🧠 Behavioral Analysis:")
                behaviors = self.analyze_behaviors()
                for species_id, behavior_data in behaviors.items():
                    species_name = self.species_populations[species_id]['config']['name']
                    print(f"   {species_name}:")
                    print(f"     Strategy: {behavior_data['dominant_strategy']}")
                    print(f"     Behaviors: {', '.join(behavior_data['behaviors'][:3])}")
                    print(f"     Diversity: {behavior_data['strategy_diversity']} strategies")
            
            # Show fitness trends
            if gen >= 2:
                print(f"\n📈 Fitness Trends:")
                for species_id, fitness_list in self.fitness_history.items():
                    if len(fitness_list) >= 3:
                        recent_trend = fitness_list[-1] - fitness_list[-3]
                        trend_symbol = "📈" if recent_trend > 0 else "📉" if recent_trend < 0 else "➡️"
                        species_name = self.species_populations[species_id]['config']['name']
                        print(f"   {species_name}: {fitness_list[-1]:.1f} {trend_symbol}")
            
            # Pause between generations
            time.sleep(1)
        
        # Final summary
        print(f"\n🎉 Evolution Complete!")
        print("=" * 50)
        print("📊 Final Results:")
        
        for species_id, species_data in self.species_populations.items():
            name = species_data['config']['name']
            final_fitness = species_data['fitness_history'][-1] if species_data['fitness_history'] else 0
            best_fitness = species_data['best_fitness']
            improvement = ((final_fitness / max(species_data['fitness_history'][0], 0.001)) - 1) * 100 if species_data['fitness_history'] else 0
            
            print(f"\n{name}:")
            print(f"  Final Fitness: {final_fitness:.1f}")
            print(f"  Best Fitness: {best_fitness:.1f}")
            print(f"  Improvement: {improvement:+.1f}%")
        
        # Final behavior analysis
        print(f"\n🧠 Final Behavioral Analysis:")
        final_behaviors = self.analyze_behaviors()
        for species_id, behavior_data in final_behaviors.items():
            species_name = self.species_populations[species_id]['config']['name']
            print(f"\n{species_name}:")
            print(f"  Evolved Strategy: {behavior_data['dominant_strategy']}")
            print(f"  Emergent Behaviors: {', '.join(behavior_data['behaviors'])}")
        
        print(f"\n✨ Neural networks have evolved new tactical behaviors!")
        print("This demonstrates how complex strategies can emerge from simple genetic operations.")

def main():
    """Main function to watch evolution"""
    print("🧬 Neural Network Evolution Watcher")
    print("=" * 50)
    print("This will show you neural networks evolving tactical behaviors in real-time!")
    print()
    
    # Ask user for number of generations
    try:
        generations = input("How many generations to evolve? (default 10): ").strip()
        generations = int(generations) if generations else 10
        generations = max(1, min(generations, 50))  # Limit to reasonable range
    except ValueError:
        generations = 10
    
    print(f"\n🎯 Running {generations} generations...")
    print("Each generation:")
    print("• Neural networks battle each other")
    print("• Best performers survive and reproduce")
    print("• Networks mutate and evolve new strategies")
    print("• Behaviors are analyzed and reported")
    
    # Create and run evolution watcher
    watcher = EvolutionWatcher()
    watcher.run_evolution(generations)

if __name__ == "__main__":
    main()

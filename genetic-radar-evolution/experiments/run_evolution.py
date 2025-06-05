"""
Main experiment runner for genetic radar evolution
"""

import numpy as np
import torch
import random
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Import our modules
import sys
sys.path.append('..')

from core.config import Config
from agents.policy_network import PolicyNetwork, SpecializedPolicyNetwork
from agents.neural_agent import NeuralAgent
from agents.behavior_analyzer import BehaviorAnalyzer, SpeciesBehaviorAnalyzer, create_behavior_report
from environment.combat_environment import CombatEnvironment
from evolution.genetic_evolution import GeneticEvolutionManager
from visualization.evolution_visualizer import EvolutionVisualizer

class GeneticRadarExperiment:
    """Main experiment class for genetic radar evolution"""
    
    def __init__(self, config: Config):
        self.config = config
        self.environment = CombatEnvironment(config)
        self.evolution_manager = GeneticEvolutionManager(config)
        self.behavior_analyzer = SpeciesBehaviorAnalyzer()
        self.visualizer = EvolutionVisualizer(config)
        
        # Experiment tracking
        self.generation = 0
        self.species_populations = {}
        self.evolution_history = []
        self.behavior_reports = []
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
    
    def initialize_species(self) -> None:
        """Initialize starting species with different neural network architectures"""
        
        species_configs = [
            {
                'id': 'alpha',
                'name': 'Alpha Hunters',
                'color': 'red',
                'strategy': 'aggressive',
                'population_size': self.config.population_per_species
            },
            {
                'id': 'beta', 
                'name': 'Beta Defenders',
                'color': 'blue',
                'strategy': 'defensive',
                'population_size': self.config.population_per_species
            },
            {
                'id': 'gamma',
                'name': 'Gamma Scouts',
                'color': 'green', 
                'strategy': 'scout',
                'population_size': self.config.population_per_species
            }
        ]
        
        for species_config in species_configs:
            population = []
            
            for i in range(species_config['population_size']):
                # Create specialized neural network
                policy = SpecializedPolicyNetwork(
                    strategy=species_config['strategy'],
                    hidden_size=self.config.hidden_size
                )
                
                agent = NeuralAgent(
                    agent_id=f"{species_config['id']}_{i}",
                    species_id=species_config['id'],
                    policy_network=policy
                )
                
                population.append(agent)
            
            self.species_populations[species_config['id']] = {
                'agents': population,
                'config': species_config,
                'fitness_history': [],
                'best_networks': []
            }
        
        print(f"🧬 Initialized {len(species_configs)} species with {self.config.population_per_species} agents each")
    
    def run_generation(self) -> Dict[str, Any]:
        """Run one generation of evolution"""
        print(f"\n🧬 Generation {self.generation + 1}")
        print("-" * 40)
        
        generation_results = {
            'generation': self.generation,
            'species_fitness': {},
            'species_behaviors': {},
            'battle_results': [],
            'emergent_behaviors': []
        }
        
        # Run multiple battles for this generation
        for battle_num in range(self.config.battles_per_generation):
            print(f"⚔️  Battle {battle_num + 1}/{self.config.battles_per_generation}")
            
            # Prepare agents for battle
            all_agents = []
            for species_id, species_data in self.species_populations.items():
                all_agents.extend(species_data['agents'])
            
            # Run battle simulation
            battle_result = self.environment.run_battle(all_agents)
            generation_results['battle_results'].append(battle_result)
            
            # Collect behavioral data
            for agent in all_agents:
                if agent.health > 0:  # Only analyze surviving agents
                    agent_data = {
                        'behavioral_metrics': agent.get_behavioral_metrics(),
                        'action_history': agent.action_history,
                        'position_history': agent.position_history
                    }
                    
                    # Analyze individual behavior
                    analyzer = BehaviorAnalyzer()
                    behavior_analysis = analyzer.analyze_agent_behavior(agent_data)
                    
                    # Store emergent behaviors
                    if behavior_analysis['emergent_patterns']:
                        generation_results['emergent_behaviors'].extend(
                            behavior_analysis['emergent_patterns']
                        )
        
        # Calculate fitness for each species
        for species_id, species_data in self.species_populations.items():
            fitness = self._calculate_species_fitness(species_id, generation_results['battle_results'])
            generation_results['species_fitness'][species_id] = fitness
            species_data['fitness_history'].append(fitness)
            
            print(f"   {species_data['config']['name']}: {fitness:.2f}")
        
        # Analyze species behaviors
        for species_id, species_data in self.species_populations.items():
            agent_data_list = []
            for agent in species_data['agents']:
                agent_data = {
                    'behavioral_metrics': agent.get_behavioral_metrics(),
                    'action_history': agent.action_history,
                    'position_history': agent.position_history
                }
                agent_data_list.append(agent_data)
            
            species_behavior = self.behavior_analyzer.analyze_species_evolution(
                species_id, agent_data_list
            )
            generation_results['species_behaviors'][species_id] = species_behavior
        
        # Evolve species based on performance
        self._evolve_species(generation_results['species_fitness'])
        
        # Store generation results
        self.evolution_history.append(generation_results)
        self.generation += 1
        
        return generation_results
    
    def _calculate_species_fitness(self, species_id: str, battle_results: List[Dict]) -> float:
        """Calculate fitness score for a species"""
        total_fitness = 0.0
        agent_count = 0
        
        for battle_result in battle_results:
            for agent_result in battle_result['agent_results']:
                if agent_result['species_id'] == species_id:
                    # Fitness components
                    survival_bonus = 100 if agent_result['survived'] else 0
                    kill_bonus = agent_result['kills'] * 50
                    damage_bonus = agent_result['damage_dealt'] * 0.5
                    survival_time_bonus = agent_result['survival_time'] * 0.1
                    
                    agent_fitness = survival_bonus + kill_bonus + damage_bonus + survival_time_bonus
                    total_fitness += agent_fitness
                    agent_count += 1
        
        return total_fitness / max(agent_count, 1)
    
    def _evolve_species(self, species_fitness: Dict[str, float]) -> None:
        """Evolve each species based on fitness"""
        for species_id, species_data in self.species_populations.items():
            fitness = species_fitness[species_id]
            
            # Determine evolution strategy based on performance
            avg_fitness = np.mean(species_data['fitness_history'][-5:]) if len(species_data['fitness_history']) >= 5 else fitness
            
            if fitness < avg_fitness * 0.7:
                # Poor performance - major evolution
                self._major_evolution(species_id)
                print(f"   🧬 Major evolution for {species_data['config']['name']}")
            elif fitness > avg_fitness * 1.3:
                # Good performance - minor refinement
                self._minor_evolution(species_id)
                print(f"   🔧 Minor evolution for {species_data['config']['name']}")
            else:
                # Average performance - random mutations
                self._random_mutations(species_id)
                print(f"   🎲 Random mutations for {species_data['config']['name']}")
    
    def _major_evolution(self, species_id: str) -> None:
        """Major evolutionary changes for struggling species"""
        species_data = self.species_populations[species_id]
        agents = species_data['agents']
        
        # Keep top 20% performers
        elite_count = max(1, int(len(agents) * 0.2))
        
        # Sort by individual fitness (simplified)
        agents.sort(key=lambda a: a.kills * 50 + a.damage_dealt * 0.5 + (100 if a.health > 0 else 0), reverse=True)
        elite_agents = agents[:elite_count]
        
        # Generate new population
        new_agents = []
        
        # Keep elite
        for agent in elite_agents:
            new_agents.append(agent)
        
        # Generate offspring through crossover and mutation
        while len(new_agents) < len(agents):
            parent1 = random.choice(elite_agents)
            parent2 = random.choice(elite_agents)
            
            # Create offspring network
            offspring_policy = parent1.policy.crossover(parent2.policy, crossover_rate=0.7)
            offspring_policy = offspring_policy.mutate(mutation_rate=0.3, mutation_strength=0.4)
            
            # Create new agent
            new_agent = NeuralAgent(
                agent_id=f"{species_id}_{len(new_agents)}",
                species_id=species_id,
                policy_network=offspring_policy
            )
            new_agents.append(new_agent)
        
        species_data['agents'] = new_agents
    
    def _minor_evolution(self, species_id: str) -> None:
        """Minor refinements for successful species"""
        species_data = self.species_populations[species_id]
        agents = species_data['agents']
        
        # Apply small mutations to all agents
        for agent in agents:
            agent.policy = agent.policy.mutate(mutation_rate=0.1, mutation_strength=0.1)
    
    def _random_mutations(self, species_id: str) -> None:
        """Random mutations for average performers"""
        species_data = self.species_populations[species_id]
        agents = species_data['agents']
        
        # Apply random mutations to subset of population
        mutation_count = int(len(agents) * 0.3)
        agents_to_mutate = random.sample(agents, mutation_count)
        
        for agent in agents_to_mutate:
            agent.policy = agent.policy.mutate(mutation_rate=0.15, mutation_strength=0.2)
    
    def run_experiment(self) -> None:
        """Run the complete genetic evolution experiment"""
        print("🚀 Starting Genetic Radar Evolution Experiment")
        print("=" * 60)
        
        # Initialize
        self.initialize_species()
        
        # Create output directory
        output_dir = Path(f"../data/experiment_{int(time.time())}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run generations
        for gen in range(self.config.num_generations):
            generation_result = self.run_generation()
            
            # Create behavior report every 5 generations
            if gen % 5 == 0:
                behavior_report = create_behavior_report(generation_result['species_behaviors'])
                self.behavior_reports.append(behavior_report)
                print(f"\n📊 Behavior Analysis (Generation {gen + 1}):")
                print(behavior_report)
            
            # Save progress
            if gen % 10 == 0:
                self._save_progress(output_dir, gen)
        
        # Final analysis and visualization
        self._create_final_report(output_dir)
        print(f"\n✅ Experiment completed! Results saved to {output_dir}")
    
    def _save_progress(self, output_dir: Path, generation: int) -> None:
        """Save experiment progress"""
        # Save evolution history
        with open(output_dir / f"evolution_history_gen_{generation}.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_copy = []
            for gen_data in self.evolution_history:
                gen_copy = gen_data.copy()
                # Remove non-serializable data
                if 'battle_results' in gen_copy:
                    del gen_copy['battle_results']
                history_copy.append(gen_copy)
            json.dump(history_copy, f, indent=2)
        
        # Save best networks
        for species_id, species_data in self.species_populations.items():
            best_agent = max(species_data['agents'], 
                           key=lambda a: a.kills * 50 + a.damage_dealt * 0.5)
            torch.save(best_agent.policy.state_dict(), 
                      output_dir / f"best_{species_id}_gen_{generation}.pt")
    
    def _create_final_report(self, output_dir: Path) -> None:
        """Create final experiment report"""
        # Generate visualizations
        self.visualizer.plot_evolution_progress(self.evolution_history, output_dir)
        self.visualizer.plot_behavior_evolution(self.behavior_analyzer.species_data, output_dir)
        
        # Create comprehensive report
        final_report = "🧬 GENETIC RADAR EVOLUTION - FINAL REPORT\n"
        final_report += "=" * 60 + "\n\n"
        
        final_report += f"Experiment Duration: {self.config.num_generations} generations\n"
        final_report += f"Battles per Generation: {self.config.battles_per_generation}\n"
        final_report += f"Population per Species: {self.config.population_per_species}\n\n"
        
        # Final species performance
        final_report += "📊 FINAL SPECIES PERFORMANCE:\n"
        final_report += "-" * 30 + "\n"
        
        for species_id, species_data in self.species_populations.items():
            final_fitness = species_data['fitness_history'][-1] if species_data['fitness_history'] else 0
            avg_fitness = np.mean(species_data['fitness_history']) if species_data['fitness_history'] else 0
            
            final_report += f"{species_data['config']['name']}:\n"
            final_report += f"  Final Fitness: {final_fitness:.2f}\n"
            final_report += f"  Average Fitness: {avg_fitness:.2f}\n"
            final_report += f"  Improvement: {((final_fitness / max(species_data['fitness_history'][0], 0.001)) - 1) * 100:.1f}%\n\n"
        
        # Emergent behaviors summary
        all_behaviors = set()
        for gen_data in self.evolution_history:
            all_behaviors.update(gen_data.get('emergent_behaviors', []))
        
        final_report += "🚀 DISCOVERED EMERGENT BEHAVIORS:\n"
        final_report += "-" * 30 + "\n"
        for behavior in sorted(all_behaviors):
            final_report += f"• {behavior.replace('_', ' ').title()}\n"
        
        # Save final report
        with open(output_dir / "final_report.txt", 'w') as f:
            f.write(final_report)
        
        print(final_report)

def main():
    """Main function to run the experiment"""
    
    # Choose experiment configuration
    print("🧬 Genetic Radar Evolution Experiment")
    print("Choose experiment type:")
    print("1. Fast experiment (20 generations, 3 battles each)")
    print("2. Detailed experiment (50 generations, 5 battles each)")
    print("3. Custom configuration")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        config = Config.create_fast_experiment()
    elif choice == "2":
        config = Config.create_detailed_experiment()
    else:
        config = Config()  # Default configuration
    
    print(f"\n📋 Experiment Configuration:")
    print(f"   Generations: {config.num_generations}")
    print(f"   Battles per generation: {config.battles_per_generation}")
    print(f"   Population per species: {config.population_per_species}")
    print(f"   Mutation rate: {config.mutation_rate}")
    print(f"   Behavior tracking: {config.behavior_tracking}")
    
    # Run experiment
    experiment = GeneticRadarExperiment(config)
    experiment.run_experiment()

if __name__ == "__main__":
    main()

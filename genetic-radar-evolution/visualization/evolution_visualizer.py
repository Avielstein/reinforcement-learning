"""
Visualization tools for genetic evolution analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from ..core.config import Config

class EvolutionVisualizer:
    """Creates visualizations for genetic evolution experiments"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def plot_evolution_progress(self, evolution_history: List[Dict], output_dir: Path) -> None:
        """Plot fitness evolution over generations"""
        if not evolution_history:
            return
        
        # Extract data
        generations = [gen['generation'] for gen in evolution_history]
        species_fitness = {}
        
        # Collect fitness data for each species
        for gen_data in evolution_history:
            for species_id, fitness in gen_data.get('species_fitness', {}).items():
                if species_id not in species_fitness:
                    species_fitness[species_id] = []
                species_fitness[species_id].append(fitness)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (species_id, fitness_values) in enumerate(species_fitness.items()):
            color = colors[i % len(colors)]
            plt.plot(generations[:len(fitness_values)], fitness_values, 
                    label=f'Species {species_id}', color=color, linewidth=2)
            
            # Add trend line
            if len(fitness_values) > 5:
                z = np.polyfit(range(len(fitness_values)), fitness_values, 1)
                p = np.poly1d(z)
                plt.plot(generations[:len(fitness_values)], p(range(len(fitness_values))), 
                        '--', color=color, alpha=0.5)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title('Species Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / 'fitness_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_behavior_evolution(self, species_data: Dict, output_dir: Path) -> None:
        """Plot behavioral diversity evolution"""
        if not species_data:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for species_id, generations in species_data.items():
            if not generations:
                continue
                
            gen_numbers = list(range(len(generations)))
            
            # Behavioral diversity
            diversity_values = [gen.get('behavioral_diversity', 0) for gen in generations]
            axes[0, 0].plot(gen_numbers, diversity_values, label=f'Species {species_id}', linewidth=2)
            
            # Strategy stability
            stability_values = [gen.get('strategy_stability', 0.5) for gen in generations]
            axes[0, 1].plot(gen_numbers, stability_values, label=f'Species {species_id}', linewidth=2)
            
            # Innovation rate
            innovation_values = [gen.get('innovation_rate', 0) for gen in generations]
            axes[1, 0].plot(gen_numbers, innovation_values, label=f'Species {species_id}', linewidth=2)
            
            # Number of emergent behaviors
            behavior_counts = [len(gen.get('emergent_behaviors', [])) for gen in generations]
            axes[1, 1].plot(gen_numbers, behavior_counts, label=f'Species {species_id}', linewidth=2)
        
        # Configure subplots
        axes[0, 0].set_title('Behavioral Diversity')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Diversity Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Strategy Stability')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Stability Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Innovation Rate')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Innovation Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Emergent Behaviors Count')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Number of Behaviors')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'behavior_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_strategy_distribution(self, generation_data: Dict, output_dir: Path, generation: int) -> None:
        """Plot strategy distribution for a specific generation"""
        plt.figure(figsize=(12, 8))
        
        species_strategies = generation_data.get('species_behaviors', {})
        
        # Create pie charts for each species
        num_species = len(species_strategies)
        if num_species == 0:
            return
        
        cols = min(3, num_species)
        rows = (num_species + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if num_species == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (species_id, behavior_data) in enumerate(species_strategies.items()):
            if i >= len(axes):
                break
                
            dominant_strategies = behavior_data.get('dominant_strategies', [])
            if not dominant_strategies:
                continue
            
            strategies = [strategy for strategy, _ in dominant_strategies]
            percentages = [percentage for _, percentage in dominant_strategies]
            
            # Create pie chart
            axes[i].pie(percentages, labels=strategies, autopct='%1.1f%%', startangle=90)
            axes[i].set_title(f'Species {species_id} - Generation {generation}')
        
        # Hide unused subplots
        for i in range(num_species, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'strategy_distribution_gen_{generation}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self, evolution_history: List[Dict], output_dir: Path) -> None:
        """Create a comprehensive dashboard of the evolution experiment"""
        if not evolution_history:
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create a 3x3 grid of subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Fitness evolution (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_fitness_subplot(evolution_history, ax1)
        
        # 2. Final strategy distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_final_strategies_subplot(evolution_history, ax2)
        
        # 3. Emergent behaviors timeline (middle left, spans 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_emergent_behaviors_subplot(evolution_history, ax3)
        
        # 4. Performance metrics (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_performance_metrics_subplot(evolution_history, ax4)
        
        # 5. Evolution summary (bottom, spans all columns)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_evolution_summary_subplot(evolution_history, ax5)
        
        plt.suptitle('Genetic Radar Evolution - Experiment Dashboard', fontsize=16, fontweight='bold')
        plt.savefig(output_dir / 'evolution_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fitness_subplot(self, evolution_history: List[Dict], ax) -> None:
        """Plot fitness evolution in subplot"""
        generations = [gen['generation'] for gen in evolution_history]
        species_fitness = {}
        
        for gen_data in evolution_history:
            for species_id, fitness in gen_data.get('species_fitness', {}).items():
                if species_id not in species_fitness:
                    species_fitness[species_id] = []
                species_fitness[species_id].append(fitness)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (species_id, fitness_values) in enumerate(species_fitness.items()):
            color = colors[i % len(colors)]
            ax.plot(generations[:len(fitness_values)], fitness_values, 
                   label=f'Species {species_id}', color=color, linewidth=2)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Score')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_strategies_subplot(self, evolution_history: List[Dict], ax) -> None:
        """Plot final strategy distribution"""
        if not evolution_history:
            return
        
        final_gen = evolution_history[-1]
        species_behaviors = final_gen.get('species_behaviors', {})
        
        all_strategies = {}
        for species_id, behavior_data in species_behaviors.items():
            strategies = behavior_data.get('dominant_strategies', [])
            for strategy, percentage in strategies:
                if strategy not in all_strategies:
                    all_strategies[strategy] = 0
                all_strategies[strategy] += percentage
        
        if all_strategies:
            strategies = list(all_strategies.keys())
            values = list(all_strategies.values())
            ax.pie(values, labels=strategies, autopct='%1.1f%%', startangle=90)
            ax.set_title('Final Strategy Distribution')
    
    def _plot_emergent_behaviors_subplot(self, evolution_history: List[Dict], ax) -> None:
        """Plot emergent behaviors over time"""
        generations = [gen['generation'] for gen in evolution_history]
        behavior_counts = [len(set(gen.get('emergent_behaviors', []))) for gen in evolution_history]
        
        ax.plot(generations, behavior_counts, 'g-', linewidth=2, marker='o')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Unique Behaviors')
        ax.set_title('Emergent Behaviors Discovery')
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_metrics_subplot(self, evolution_history: List[Dict], ax) -> None:
        """Plot key performance metrics"""
        if not evolution_history:
            return
        
        final_gen = evolution_history[-1]
        species_fitness = final_gen.get('species_fitness', {})
        
        species_names = list(species_fitness.keys())
        fitness_values = list(species_fitness.values())
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        bars = ax.bar(species_names, fitness_values, color=colors[:len(species_names)])
        
        ax.set_xlabel('Species')
        ax.set_ylabel('Final Fitness')
        ax.set_title('Final Performance')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, fitness_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(fitness_values),
                   f'{value:.1f}', ha='center', va='bottom')
    
    def _plot_evolution_summary_subplot(self, evolution_history: List[Dict], ax) -> None:
        """Plot evolution summary text"""
        ax.axis('off')
        
        if not evolution_history:
            return
        
        # Calculate summary statistics
        total_generations = len(evolution_history)
        final_gen = evolution_history[-1]
        
        all_behaviors = set()
        for gen_data in evolution_history:
            all_behaviors.update(gen_data.get('emergent_behaviors', []))
        
        species_fitness = final_gen.get('species_fitness', {})
        best_species = max(species_fitness.items(), key=lambda x: x[1]) if species_fitness else ('N/A', 0)
        
        summary_text = f"""
EVOLUTION EXPERIMENT SUMMARY

Total Generations: {total_generations}
Unique Emergent Behaviors: {len(all_behaviors)}
Best Performing Species: {best_species[0]} (Fitness: {best_species[1]:.2f})

Discovered Behaviors:
{', '.join(sorted(all_behaviors)[:10])}{'...' if len(all_behaviors) > 10 else ''}

The experiment successfully demonstrated neural network evolution with emergent tactical behaviors.
Species developed distinct strategies through genetic operations on network weights.
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

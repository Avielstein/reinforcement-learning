"""
Simple demo of the Multi-Agent Genetic Team Survival System
"""

import sys
import os

# Add the current directory to the path so we can import modules
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

# Import modules directly
from core.config import Config
from simulation.episode_runner import EpisodeRunner

def run_simple_demo():
    """Run a simple demonstration of the system"""
    print("🧬 Multi-Agent Genetic Team Survival System")
    print("=" * 50)
    
    # Create configuration
    config = Config()
    
    # Adjust config for quick demo
    config.INITIAL_TEAMS = 3
    config.STARTING_TEAM_SIZE = 3
    config.EPISODE_LENGTH = 200  # Shorter episodes for demo
    config.MAX_TEAM_SIZE = 8
    
    print(f"Configuration:")
    print(f"  Initial Teams: {config.INITIAL_TEAMS}")
    print(f"  Starting Team Size: {config.STARTING_TEAM_SIZE}")
    print(f"  Episode Length: {config.EPISODE_LENGTH}")
    print(f"  World Size: {config.WORLD_WIDTH}x{config.WORLD_HEIGHT}")
    print()
    
    # Create episode runner
    runner = EpisodeRunner(config)
    
    # Add callbacks for real-time updates
    def episode_callback(episode_result, population_state):
        print(f"🏆 Episode {episode_result.episode_id} Results:")
        for team_id, survivors in episode_result.team_survivors.items():
            initial = episode_result.team_initial_sizes[team_id]
            rate = survivors / initial if initial > 0 else 0
            print(f"   Team {team_id}: {survivors}/{initial} survived ({rate:.1%})")
        
        if episode_result.team_eliminations:
            print(f"   💀 Eliminated: Teams {episode_result.team_eliminations}")
        print()
    
    runner.add_episode_callback(episode_callback)
    
    # Run demo episodes
    print("🚀 Starting simulation...")
    print("Running 5 episodes to demonstrate evolution...")
    print()
    
    try:
        # Run 5 episodes
        runner.start_simulation(max_episodes=5, background=False)
        
        # Get final statistics
        print("📊 Final Statistics:")
        print("=" * 30)
        
        performance = runner.get_performance_summary()
        
        # Population stats
        pop_stats = performance['population']
        print(f"Population:")
        print(f"  Active Teams: {pop_stats['total_teams']}")
        print(f"  Total Agents: {pop_stats['total_agents']}")
        print(f"  Alive Agents: {pop_stats['alive_agents']}")
        print(f"  Average Team Size: {pop_stats['average_team_size']:.1f}")
        print(f"  Average Survival Rate: {pop_stats['average_survival_rate']:.1%}")
        print()
        
        # Evolution insights
        evolution = performance['evolution']
        if evolution:
            print(f"Evolution:")
            print(f"  Growing Teams: {evolution['growing_teams']}")
            print(f"  Declining Teams: {evolution['declining_teams']}")
            print(f"  Stable Teams: {evolution['stable_teams']}")
            print(f"  Generation Spread: {evolution['generation_spread']}")
            print(f"  Average Diversity: {evolution['average_diversity']:.3f}")
            print()
        
        # Simulation stats
        sim_stats = performance['simulation']
        if sim_stats:
            print(f"Simulation:")
            print(f"  Total Episodes: {sim_stats['total_episodes']}")
            print(f"  Total Time: {sim_stats['total_simulation_time']:.1f}s")
            print(f"  Episodes/min: {sim_stats['episodes_per_minute']:.1f}")
            print()
        
        print("🎉 Demo completed successfully!")
        print()
        print("What you just saw:")
        print("• Teams of agents competing for survival")
        print("• Genetic evolution of neural network policies")
        print("• Dynamic team sizes (growth, shrinking, splitting)")
        print("• Emergent behaviors through pure survival pressure")
        print()
        print("To see more detailed evolution, run longer simulations!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        runner.stop_simulation()
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        runner.stop_simulation()

if __name__ == "__main__":
    run_simple_demo()

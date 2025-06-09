"""
Compare different TD learning methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time
import os

from ..config.environment import EnvironmentConfig
from ..config.td_config import TDConfig
from ..config.training import TrainingConfig
from ..core.environment import TDFishEnvironment
from ..core.agent import TDFishAgent


def run_td_comparison(num_episodes: int = 500, pattern: str = 'random_walk') -> Dict[str, Any]:
    """
    Compare different TD learning methods on the fish following task.
    
    Args:
        num_episodes: Number of training episodes
        pattern: Target movement pattern to use
        
    Returns:
        Dictionary containing results for each method
    """
    print("🧠 TD Learning Methods Comparison")
    print("=" * 50)
    
    # Create configurations
    env_config = EnvironmentConfig()
    training_config = TrainingConfig()
    
    # TD methods to compare
    td_methods = ['td_0', 'td_lambda', 'n_step_td']
    results = {}
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    for method in td_methods:
        print(f"\n🔬 Testing {method.upper()}")
        print("-" * 30)
        
        # Create TD config for this method
        td_config = TDConfig()
        td_config.method = method
        
        # Method-specific parameters
        if method == 'td_lambda':
            td_config.lambda_param = 0.9
        elif method == 'n_step_td':
            td_config.n_steps = 5
        
        # Create environment and agent
        env = TDFishEnvironment(env_config, pattern)
        agent = TDFishAgent(env_config, td_config, device)
        
        # Training metrics
        episode_rewards = []
        episode_distances = []
        td_errors = []
        training_times = []
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment and agent
            obs = env.reset()
            agent.policy.reset_action_smoothing()
            
            episode_reward = 0
            distances = []
            episode_td_errors = []
            done = False
            
            while not done:
                # Select action
                action = agent.select_action(obs)
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                
                # Update agent
                stats = agent.update(obs, action, reward, next_obs, done)
                
                # Track metrics
                episode_reward += reward
                distances.append(info['distance_to_target'])
                if 'td_error' in stats:
                    episode_td_errors.append(abs(stats['td_error']))
                
                obs = next_obs
            
            # End episode
            avg_distance = np.mean(distances)
            agent.end_episode(episode_reward, avg_distance)
            
            # Store metrics
            episode_rewards.append(episode_reward)
            episode_distances.append(avg_distance)
            if episode_td_errors:
                td_errors.append(np.mean(episode_td_errors))
            
            # Progress reporting
            if (episode + 1) % 50 == 0:
                recent_reward = np.mean(episode_rewards[-10:])
                recent_distance = np.mean(episode_distances[-10:])
                print(f"Episode {episode + 1:3d}: Reward={recent_reward:6.2f}, Distance={recent_distance:5.1f}")
        
        training_time = time.time() - start_time
        training_times.append(training_time)
        
        # Final evaluation
        print(f"\n📊 Final Evaluation for {method.upper()}")
        eval_results = agent.evaluate(env, num_episodes=20)
        
        print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"Average Distance: {eval_results['avg_distance']:.1f} ± {eval_results['std_distance']:.1f}")
        print(f"Success Rate: {eval_results['success_rate']:.1%}")
        print(f"Training Time: {training_time:.1f}s")
        
        # Store results
        results[method] = {
            'episode_rewards': episode_rewards,
            'episode_distances': episode_distances,
            'td_errors': td_errors,
            'training_time': training_time,
            'eval_results': eval_results,
            'final_stats': agent.get_stats()
        }
    
    # Create comparison plots
    _plot_comparison_results(results, pattern)
    
    # Print summary
    _print_comparison_summary(results)
    
    return results


def _plot_comparison_results(results: Dict[str, Any], pattern: str):
    """Create comparison plots for different TD methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'TD Learning Methods Comparison - {pattern.title()} Pattern', fontsize=16)
    
    colors = {'td_0': 'blue', 'td_lambda': 'red', 'n_step_td': 'green'}
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    for method, data in results.items():
        rewards = data['episode_rewards']
        # Smooth with moving average
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, 
                label=method.upper(), color=colors[method], linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards (20-episode moving average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Distances
    ax2 = axes[0, 1]
    for method, data in results.items():
        distances = data['episode_distances']
        # Smooth with moving average
        smoothed = np.convolve(distances, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(distances)), smoothed, 
                label=method.upper(), color=colors[method], linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Distance to Target')
    ax2.set_title('Following Performance (20-episode moving average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: TD Errors
    ax3 = axes[1, 0]
    for method, data in results.items():
        if data['td_errors']:
            td_errors = data['td_errors']
            smoothed = np.convolve(td_errors, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(td_errors)), smoothed, 
                    label=method.upper(), color=colors[method], linewidth=2)
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average TD Error')
    ax3.set_title('TD Error Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Final Performance Comparison
    ax4 = axes[1, 1]
    methods = list(results.keys())
    final_rewards = [results[m]['eval_results']['avg_reward'] for m in methods]
    final_distances = [results[m]['eval_results']['avg_distance'] for m in methods]
    success_rates = [results[m]['eval_results']['success_rate'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x - width, final_rewards, width, label='Avg Reward', alpha=0.8)
    bars2 = ax4.bar(x, final_distances, width, label='Avg Distance', alpha=0.8)
    bars3 = ax4_twin.bar(x + width, success_rates, width, label='Success Rate', alpha=0.8, color='orange')
    
    ax4.set_xlabel('TD Method')
    ax4.set_ylabel('Reward / Distance')
    ax4_twin.set_ylabel('Success Rate')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.upper() for m in methods])
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/td_comparison_{pattern}.png', dpi=300, bbox_inches='tight')
    plt.show()


def _print_comparison_summary(results: Dict[str, Any]):
    """Print summary comparison of TD methods."""
    print("\n" + "="*60)
    print("📈 COMPARISON SUMMARY")
    print("="*60)
    
    # Create comparison table
    methods = list(results.keys())
    
    print(f"{'Method':<12} {'Avg Reward':<12} {'Avg Distance':<13} {'Success Rate':<13} {'Training Time':<13}")
    print("-" * 65)
    
    for method in methods:
        eval_results = results[method]['eval_results']
        training_time = results[method]['training_time']
        
        print(f"{method.upper():<12} "
              f"{eval_results['avg_reward']:<12.2f} "
              f"{eval_results['avg_distance']:<13.1f} "
              f"{eval_results['success_rate']:<13.1%} "
              f"{training_time:<13.1f}s")
    
    # Find best performing method
    best_reward_method = max(methods, key=lambda m: results[m]['eval_results']['avg_reward'])
    best_distance_method = min(methods, key=lambda m: results[m]['eval_results']['avg_distance'])
    best_success_method = max(methods, key=lambda m: results[m]['eval_results']['success_rate'])
    fastest_method = min(methods, key=lambda m: results[m]['training_time'])
    
    print("\n🏆 WINNERS:")
    print(f"Best Reward:      {best_reward_method.upper()}")
    print(f"Best Following:   {best_distance_method.upper()}")
    print(f"Best Success:     {best_success_method.upper()}")
    print(f"Fastest Training: {fastest_method.upper()}")
    
    # Learning efficiency analysis
    print("\n📊 LEARNING EFFICIENCY:")
    for method in methods:
        rewards = results[method]['episode_rewards']
        # Calculate episodes to reach 75% of final performance
        final_performance = np.mean(rewards[-50:])
        target_performance = final_performance * 0.75
        
        episodes_to_target = len(rewards)
        for i, reward in enumerate(rewards):
            if np.mean(rewards[max(0, i-10):i+1]) >= target_performance:
                episodes_to_target = i
                break
        
        print(f"{method.upper():<12}: {episodes_to_target} episodes to reach 75% performance")


if __name__ == "__main__":
    # Run comparison with different patterns
    patterns = ['random_walk', 'circular', 'figure8']
    
    for pattern in patterns:
        print(f"\n{'='*20} {pattern.upper()} PATTERN {'='*20}")
        results = run_td_comparison(num_episodes=300, pattern=pattern)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        torch.save(results, f'results/td_comparison_{pattern}_results.pt')

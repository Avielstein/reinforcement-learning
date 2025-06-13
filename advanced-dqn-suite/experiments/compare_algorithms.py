"""
Algorithm Comparison Framework
Compare DQN variants across multiple environments with statistical rigor
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import torch
import time
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.base_dqn import BaseDQNAgent
from algorithms.double_dqn import DoubleDQNAgent


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    env_name: str
    num_episodes: int
    num_seeds: int
    eval_frequency: int
    eval_episodes: int
    save_results: bool = True
    plot_results: bool = True


class AlgorithmComparison:
    """Framework for comparing DQN algorithms"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        
        # Create results directory
        self.results_dir = Path("results") / config.env_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_agent(self, algorithm: str, env, seed: int, **kwargs) -> BaseDQNAgent:
        """Create agent based on algorithm name"""
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        if algorithm == "DQN":
            return BaseDQNAgent(obs_dim, act_dim, **kwargs)
        elif algorithm == "DoubleDQN":
            return DoubleDQNAgent(obs_dim, act_dim, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train_agent(self, agent: BaseDQNAgent, env, seed: int) -> Dict[str, List[float]]:
        """Train a single agent and return training metrics"""
        print(f"Training {agent.__class__.__name__} with seed {seed}")
        
        # Set environment seed
        env.reset(seed=seed)
        
        training_rewards = []
        evaluation_rewards = []
        evaluation_episodes = []
        
        for episode in range(self.config.num_episodes):
            # Train one episode
            episode_reward, episode_length = agent.train_episode(env)
            training_rewards.append(episode_reward)
            
            # Evaluate periodically
            if episode % self.config.eval_frequency == 0:
                eval_results = agent.evaluate(env, self.config.eval_episodes)
                evaluation_rewards.append(eval_results['mean_reward'])
                evaluation_episodes.append(episode)
                
                print(f"Episode {episode}: Train Reward = {episode_reward:.2f}, "
                      f"Eval Reward = {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        
        return {
            'training_rewards': training_rewards,
            'evaluation_rewards': evaluation_rewards,
            'evaluation_episodes': evaluation_episodes,
            'final_eval': agent.evaluate(env, self.config.eval_episodes * 2)  # More episodes for final eval
        }
    
    def run_comparison(self, algorithms: List[str], **agent_kwargs) -> Dict[str, Any]:
        """Run comparison across multiple algorithms and seeds"""
        print(f"Running comparison on {self.config.env_name}")
        print(f"Algorithms: {algorithms}")
        print(f"Seeds: {self.config.num_seeds}")
        print(f"Episodes per run: {self.config.num_episodes}")
        
        results = {}
        
        for algorithm in algorithms:
            print(f"\n{'='*50}")
            print(f"Training {algorithm}")
            print(f"{'='*50}")
            
            algorithm_results = {
                'training_rewards': [],
                'evaluation_rewards': [],
                'evaluation_episodes': [],
                'final_evaluations': [],
                'training_times': []
            }
            
            for seed in range(self.config.num_seeds):
                # Create environment
                env = gym.make(self.config.env_name)
                
                # Create agent
                agent = self.create_agent(algorithm, env, seed, **agent_kwargs)
                
                # Train agent
                start_time = time.time()
                seed_results = self.train_agent(agent, env, seed)
                training_time = time.time() - start_time
                
                # Store results
                algorithm_results['training_rewards'].append(seed_results['training_rewards'])
                algorithm_results['evaluation_rewards'].append(seed_results['evaluation_rewards'])
                algorithm_results['evaluation_episodes'] = seed_results['evaluation_episodes']  # Same for all seeds
                algorithm_results['final_evaluations'].append(seed_results['final_eval']['mean_reward'])
                algorithm_results['training_times'].append(training_time)
                
                env.close()
                
                print(f"Seed {seed} completed in {training_time:.1f}s. "
                      f"Final performance: {seed_results['final_eval']['mean_reward']:.2f}")
            
            results[algorithm] = algorithm_results
        
        self.results = results
        
        if self.config.save_results:
            self.save_results()
        
        if self.config.plot_results:
            self.plot_comparison()
        
        return results
    
    def save_results(self):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for algorithm, data in self.results.items():
            serializable_results[algorithm] = {
                'training_rewards': [list(rewards) for rewards in data['training_rewards']],
                'evaluation_rewards': [list(rewards) for rewards in data['evaluation_rewards']],
                'evaluation_episodes': list(data['evaluation_episodes']),
                'final_evaluations': list(data['final_evaluations']),
                'training_times': list(data['training_times'])
            }
        
        # Save experiment config and results
        save_data = {
            'config': {
                'env_name': self.config.env_name,
                'num_episodes': self.config.num_episodes,
                'num_seeds': self.config.num_seeds,
                'eval_frequency': self.config.eval_frequency,
                'eval_episodes': self.config.eval_episodes
            },
            'results': serializable_results
        }
        
        filepath = self.results_dir / f"comparison_results_{int(time.time())}.json"
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def plot_comparison(self):
        """Create comprehensive comparison plots"""
        if not self.results:
            print("No results to plot!")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'DQN Algorithm Comparison - {self.config.env_name}', fontsize=16)
        
        algorithms = list(self.results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        # 1. Training curves (smoothed)
        ax1 = axes[0, 0]
        for i, algorithm in enumerate(algorithms):
            training_rewards = np.array(self.results[algorithm]['training_rewards'])
            
            # Calculate mean and std across seeds
            mean_rewards = np.mean(training_rewards, axis=0)
            std_rewards = np.std(training_rewards, axis=0)
            
            # Smooth the curves
            window_size = min(50, len(mean_rewards) // 10)
            if window_size > 1:
                smoothed_mean = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
                smoothed_std = np.convolve(std_rewards, np.ones(window_size)/window_size, mode='valid')
                x = np.arange(window_size-1, len(mean_rewards))
            else:
                smoothed_mean = mean_rewards
                smoothed_std = std_rewards
                x = np.arange(len(mean_rewards))
            
            ax1.plot(x, smoothed_mean, label=algorithm, color=colors[i], linewidth=2)
            ax1.fill_between(x, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, 
                           alpha=0.3, color=colors[i])
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Training Reward')
        ax1.set_title('Training Progress (Smoothed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Evaluation curves
        ax2 = axes[0, 1]
        for i, algorithm in enumerate(algorithms):
            eval_rewards = np.array(self.results[algorithm]['evaluation_rewards'])
            eval_episodes = self.results[algorithm]['evaluation_episodes']
            
            mean_eval = np.mean(eval_rewards, axis=0)
            std_eval = np.std(eval_rewards, axis=0)
            
            ax2.plot(eval_episodes, mean_eval, label=algorithm, color=colors[i], 
                    linewidth=2, marker='o', markersize=4)
            ax2.fill_between(eval_episodes, mean_eval - std_eval, mean_eval + std_eval, 
                           alpha=0.3, color=colors[i])
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Evaluation Reward')
        ax2.set_title('Evaluation Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Final performance comparison
        ax3 = axes[1, 0]
        final_performances = []
        algorithm_names = []
        
        for algorithm in algorithms:
            final_performances.extend(self.results[algorithm]['final_evaluations'])
            algorithm_names.extend([algorithm] * len(self.results[algorithm]['final_evaluations']))
        
        # Create box plot
        data_for_box = [self.results[alg]['final_evaluations'] for alg in algorithms]
        box_plot = ax3.boxplot(data_for_box, labels=algorithms, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Final Evaluation Reward')
        ax3.set_title('Final Performance Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Training time comparison
        ax4 = axes[1, 1]
        training_times = [np.mean(self.results[alg]['training_times']) for alg in algorithms]
        training_stds = [np.std(self.results[alg]['training_times']) for alg in algorithms]
        
        bars = ax4.bar(algorithms, training_times, yerr=training_stds, 
                      color=colors, alpha=0.7, capsize=5)
        ax4.set_ylabel('Training Time (seconds)')
        ax4.set_title('Training Efficiency')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(training_stds)*0.1,
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"comparison_plot_{int(time.time())}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {plot_path}")
        
        plt.show()
    
    def print_statistical_summary(self):
        """Print statistical summary of results"""
        if not self.results:
            print("No results to summarize!")
            return
        
        print(f"\n{'='*60}")
        print(f"STATISTICAL SUMMARY - {self.config.env_name}")
        print(f"{'='*60}")
        
        for algorithm in self.results.keys():
            final_scores = self.results[algorithm]['final_evaluations']
            training_times = self.results[algorithm]['training_times']
            
            print(f"\n{algorithm}:")
            print(f"  Final Performance: {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")
            print(f"  Best Performance: {np.max(final_scores):.2f}")
            print(f"  Worst Performance: {np.min(final_scores):.2f}")
            print(f"  Training Time: {np.mean(training_times):.1f} ± {np.std(training_times):.1f} seconds")
        
        # Statistical significance test
        if len(self.results) == 2:
            algorithms = list(self.results.keys())
            scores1 = self.results[algorithms[0]]['final_evaluations']
            scores2 = self.results[algorithms[1]]['final_evaluations']
            
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(scores1, scores2)
            
            print(f"\nStatistical Test ({algorithms[0]} vs {algorithms[1]}):")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.3f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")


def main():
    """Run the comparison experiment"""
    # Configuration
    config = ExperimentConfig(
        env_name='CartPole-v1',
        num_episodes=1000,
        num_seeds=5,
        eval_frequency=100,
        eval_episodes=10
    )
    
    # Agent hyperparameters
    agent_kwargs = {
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 50000,
        'batch_size': 32,
        'target_update_freq': 500
    }
    
    # Run comparison
    comparison = AlgorithmComparison(config)
    results = comparison.run_comparison(['DQN', 'DoubleDQN'], **agent_kwargs)
    
    # Print summary
    comparison.print_statistical_summary()
    
    print(f"\nExperiment completed! Check the results directory: {comparison.results_dir}")


if __name__ == "__main__":
    main()

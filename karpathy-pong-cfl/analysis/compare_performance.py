"""
Performance Comparison Analysis for Baseline vs CFL-Enhanced Pong
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from typing import Dict, List, Tuple
import glob

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_training_stats(stats_path: str) -> Dict:
    """Load training statistics from pickle file"""
    try:
        with open(stats_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {stats_path}: {e}")
        return None

def find_latest_stats(directory: str, pattern: str) -> str:
    """Find the latest statistics file matching pattern"""
    pattern_path = os.path.join(directory, pattern)
    files = glob.glob(pattern_path)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def smooth_curve(data: List[float], window: int = 50) -> List[float]:
    """Apply moving average smoothing to data"""
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed

def analyze_learning_efficiency(reward_history: List[float], 
                              convergence_threshold: float = -15.0) -> Dict:
    """Analyze learning efficiency metrics"""
    if not reward_history:
        return {}
    
    # Find convergence point (when running average stays above threshold)
    running_avg = []
    window = 100
    for i in range(len(reward_history)):
        start = max(0, i - window + 1)
        running_avg.append(np.mean(reward_history[start:i+1]))
    
    convergence_episode = None
    for i in range(window, len(running_avg)):
        if all(avg >= convergence_threshold for avg in running_avg[i-window:i]):
            convergence_episode = i
            break
    
    # Calculate other metrics
    final_performance = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
    max_performance = max(reward_history)
    min_performance = min(reward_history)
    
    return {
        'convergence_episode': convergence_episode,
        'final_performance': final_performance,
        'max_performance': max_performance,
        'min_performance': min_performance,
        'total_episodes': len(reward_history),
        'performance_variance': np.var(reward_history[-100:]) if len(reward_history) >= 100 else np.var(reward_history)
    }

def compare_baseline_vs_cfl():
    """Main comparison function"""
    print("Pong Performance Comparison: Baseline vs CFL-Enhanced")
    print("=" * 60)
    
    # Find latest statistics files
    baseline_stats_path = find_latest_stats('../baseline/models', 'stats_*.p')
    cfl_stats_path = find_latest_stats('../cfl_enhanced/models', 'stats_cfl_*.p')
    
    if not baseline_stats_path:
        print("No baseline statistics found. Please run baseline training first.")
        return
    
    if not cfl_stats_path:
        print("No CFL statistics found. Please run CFL-enhanced training first.")
        return
    
    print(f"Loading baseline stats from: {baseline_stats_path}")
    print(f"Loading CFL stats from: {cfl_stats_path}")
    
    # Load statistics
    baseline_stats = load_training_stats(baseline_stats_path)
    cfl_stats = load_training_stats(cfl_stats_path)
    
    if not baseline_stats or not cfl_stats:
        print("Failed to load statistics files.")
        return
    
    # Extract reward histories
    baseline_rewards = baseline_stats['reward_history']
    cfl_rewards = cfl_stats['reward_history']
    
    # Analyze learning efficiency
    baseline_analysis = analyze_learning_efficiency(baseline_rewards)
    cfl_analysis = analyze_learning_efficiency(cfl_rewards)
    
    # Print comparison results
    print("\nLearning Efficiency Analysis:")
    print("-" * 40)
    
    metrics = ['final_performance', 'max_performance', 'convergence_episode', 'total_episodes', 'performance_variance']
    
    for metric in metrics:
        baseline_val = baseline_analysis.get(metric, 'N/A')
        cfl_val = cfl_analysis.get(metric, 'N/A')
        
        print(f"{metric:20s}: Baseline={baseline_val:>8}, CFL={cfl_val:>8}")
        
        if isinstance(baseline_val, (int, float)) and isinstance(cfl_val, (int, float)):
            if metric == 'convergence_episode' and baseline_val and cfl_val:
                improvement = (baseline_val - cfl_val) / baseline_val * 100
                print(f"{'':20s}  CFL converged {improvement:+.1f}% faster")
            elif metric in ['final_performance', 'max_performance']:
                improvement = (cfl_val - baseline_val) / abs(baseline_val) * 100
                print(f"{'':20s}  CFL performed {improvement:+.1f}% better")
    
    # Create comprehensive comparison plots
    create_comparison_plots(baseline_rewards, cfl_rewards, cfl_stats)
    
    # Analyze CFL-specific metrics
    if 'cfl_trained' in cfl_stats and cfl_stats['cfl_trained']:
        analyze_cfl_impact(cfl_rewards, cfl_stats)

def create_comparison_plots(baseline_rewards: List[float], 
                          cfl_rewards: List[float], 
                          cfl_stats: Dict):
    """Create comprehensive comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Raw reward curves
    axes[0, 0].plot(baseline_rewards, label='Baseline', alpha=0.7)
    axes[0, 0].plot(cfl_rewards, label='CFL-Enhanced', alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Smoothed curves
    baseline_smooth = smooth_curve(baseline_rewards, window=50)
    cfl_smooth = smooth_curve(cfl_rewards, window=50)
    
    axes[0, 1].plot(baseline_smooth, label='Baseline (smoothed)', linewidth=2)
    axes[0, 1].plot(cfl_smooth, label='CFL-Enhanced (smoothed)', linewidth=2)
    axes[0, 1].set_title('Smoothed Learning Curves')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Smoothed Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Running averages
    def running_average(data, window=100):
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]
    
    baseline_running = running_average(baseline_rewards)
    cfl_running = running_average(cfl_rewards)
    
    axes[0, 2].plot(baseline_running, label='Baseline', linewidth=2)
    axes[0, 2].plot(cfl_running, label='CFL-Enhanced', linewidth=2)
    axes[0, 2].set_title('Running Average (window=100)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Running Average Reward')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: Performance distribution
    axes[1, 0].hist(baseline_rewards, bins=50, alpha=0.7, label='Baseline', density=True)
    axes[1, 0].hist(cfl_rewards, bins=50, alpha=0.7, label='CFL-Enhanced', density=True)
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Cumulative performance
    baseline_cumsum = np.cumsum(baseline_rewards)
    cfl_cumsum = np.cumsum(cfl_rewards)
    
    axes[1, 1].plot(baseline_cumsum, label='Baseline')
    axes[1, 1].plot(cfl_cumsum, label='CFL-Enhanced')
    axes[1, 1].set_title('Cumulative Reward')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: CFL transition analysis (if applicable)
    if 'use_cfl_features' in cfl_stats and cfl_stats['use_cfl_features']:
        # Estimate transition point (assuming 500 episodes for data collection)
        transition_point = 500
        
        if len(cfl_rewards) > transition_point:
            pre_cfl = cfl_rewards[:transition_point]
            post_cfl = cfl_rewards[transition_point:]
            
            axes[1, 2].plot(range(len(pre_cfl)), pre_cfl, label='Pre-CFL (Raw Features)', alpha=0.7)
            axes[1, 2].plot(range(len(pre_cfl), len(cfl_rewards)), post_cfl, 
                           label='Post-CFL (Macro Features)', alpha=0.7)
            axes[1, 2].axvline(x=transition_point, color='red', linestyle='--', 
                              label='CFL Transition')
            axes[1, 2].set_title('CFL Transition Impact')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Reward')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
    else:
        axes[1, 2].text(0.5, 0.5, 'CFL Transition\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('CFL Transition Impact')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/baseline_vs_cfl_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plots saved to: ../results/baseline_vs_cfl_comparison.png")

def analyze_cfl_impact(cfl_rewards: List[float], cfl_stats: Dict):
    """Analyze the specific impact of CFL transition"""
    print("\nCFL Impact Analysis:")
    print("-" * 40)
    
    # Assume transition at episode 500 (data collection phase)
    transition_point = 500
    
    if len(cfl_rewards) <= transition_point:
        print("Not enough data to analyze CFL impact.")
        return
    
    pre_cfl_rewards = cfl_rewards[:transition_point]
    post_cfl_rewards = cfl_rewards[transition_point:]
    
    # Calculate metrics for each phase
    pre_cfl_avg = np.mean(pre_cfl_rewards[-50:]) if len(pre_cfl_rewards) >= 50 else np.mean(pre_cfl_rewards)
    post_cfl_avg = np.mean(post_cfl_rewards[:50]) if len(post_cfl_rewards) >= 50 else np.mean(post_cfl_rewards)
    
    pre_cfl_var = np.var(pre_cfl_rewards[-50:]) if len(pre_cfl_rewards) >= 50 else np.var(pre_cfl_rewards)
    post_cfl_var = np.var(post_cfl_rewards[:50]) if len(post_cfl_rewards) >= 50 else np.var(post_cfl_rewards)
    
    print(f"Pre-CFL Performance (Raw Features):")
    print(f"  Average Reward: {pre_cfl_avg:.2f}")
    print(f"  Variance: {pre_cfl_var:.2f}")
    
    print(f"\nPost-CFL Performance (Macro Features):")
    print(f"  Average Reward: {post_cfl_avg:.2f}")
    print(f"  Variance: {post_cfl_var:.2f}")
    
    # Calculate improvement
    performance_change = (post_cfl_avg - pre_cfl_avg) / abs(pre_cfl_avg) * 100
    variance_change = (post_cfl_var - pre_cfl_var) / pre_cfl_var * 100
    
    print(f"\nCFL Impact:")
    print(f"  Performance Change: {performance_change:+.1f}%")
    print(f"  Variance Change: {variance_change:+.1f}%")
    
    # Feature dimensionality reduction
    print(f"\nFeature Compression:")
    print(f"  Raw Features: 6400 dimensions (80x80 pixels)")
    print(f"  CFL Features: 64 dimensions")
    print(f"  Compression Ratio: 100:1")

if __name__ == "__main__":
    compare_baseline_vs_cfl()

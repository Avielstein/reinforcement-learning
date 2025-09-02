"""
Comparison Script: Baseline vs CFL-Enhanced Pong Training
Runs both approaches and compares their performance metrics

This script demonstrates the effectiveness of CFL for RL acceleration by:
1. Running baseline Pong training (raw pixels)
2. Running CFL-enhanced training (macro-states)
3. Comparing learning curves, sample efficiency, and final performance
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import subprocess
import time
from datetime import datetime
import json

class PongComparison:
    """Compare baseline vs CFL-enhanced Pong training"""
    
    def __init__(self, max_episodes=300):
        self.max_episodes = max_episodes
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_baseline(self):
        """Run baseline Pong training"""
        print("🎮 Running Baseline Pong Training...")
        print("=" * 50)
        
        # Modify baseline script to stop after max_episodes
        baseline_script = f"""
import sys
sys.path.append('.')
exec(open('baseline/train_pong.py').read().replace('while True:', 'while episode_number < {self.max_episodes}:'))
"""
        
        # Write temporary script
        with open('temp_baseline.py', 'w') as f:
            f.write(baseline_script)
        
        # Run baseline training
        start_time = time.time()
        try:
            result = subprocess.run(['python', 'temp_baseline.py'], 
                                  capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            baseline_time = time.time() - start_time
            
            if result.returncode == 0:
                print("✅ Baseline training completed successfully")
                self.results['baseline'] = {
                    'success': True,
                    'training_time': baseline_time,
                    'output': result.stdout
                }
            else:
                print(f"❌ Baseline training failed: {result.stderr}")
                self.results['baseline'] = {
                    'success': False,
                    'error': result.stderr
                }
        except subprocess.TimeoutExpired:
            print("⏰ Baseline training timed out")
            self.results['baseline'] = {
                'success': False,
                'error': 'Training timed out after 1 hour'
            }
        finally:
            # Clean up
            if os.path.exists('temp_baseline.py'):
                os.remove('temp_baseline.py')
    
    def run_cfl_enhanced(self):
        """Run CFL-enhanced Pong training"""
        print("\n🧠 Running CFL-Enhanced Pong Training...")
        print("=" * 50)
        
        # Modify CFL script to stop after max_episodes
        cfl_script = f"""
import sys
sys.path.append('.')
exec(open('train_cfl_pong.py').read().replace('while True:', 'while episode_number < {self.max_episodes}:'))
"""
        
        # Write temporary script
        with open('temp_cfl.py', 'w') as f:
            f.write(cfl_script)
        
        # Run CFL training
        start_time = time.time()
        try:
            result = subprocess.run(['python', 'temp_cfl.py'], 
                                  capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            cfl_time = time.time() - start_time
            
            if result.returncode == 0:
                print("✅ CFL-enhanced training completed successfully")
                self.results['cfl_enhanced'] = {
                    'success': True,
                    'training_time': cfl_time,
                    'output': result.stdout
                }
            else:
                print(f"❌ CFL-enhanced training failed: {result.stderr}")
                self.results['cfl_enhanced'] = {
                    'success': False,
                    'error': result.stderr
                }
        except subprocess.TimeoutExpired:
            print("⏰ CFL-enhanced training timed out")
            self.results['cfl_enhanced'] = {
                'success': False,
                'error': 'Training timed out after 1 hour'
            }
        finally:
            # Clean up
            if os.path.exists('temp_cfl.py'):
                os.remove('temp_cfl.py')
    
    def load_training_stats(self):
        """Load training statistics from saved files"""
        print("\n📊 Loading training statistics...")
        
        # Find most recent stats files
        baseline_stats = self._find_latest_stats('baseline')
        cfl_stats = self._find_latest_stats('cfl')
        
        if baseline_stats:
            self.results['baseline']['stats'] = baseline_stats
            print(f"✅ Loaded baseline stats: {len(baseline_stats['reward_history'])} episodes")
        
        if cfl_stats:
            self.results['cfl_enhanced']['stats'] = cfl_stats
            print(f"✅ Loaded CFL stats: {len(cfl_stats['reward_history'])} episodes")
    
    def _find_latest_stats(self, approach):
        """Find the most recent stats file for an approach"""
        if not os.path.exists('models'):
            return None
            
        stats_files = [f for f in os.listdir('models') if f.startswith(f'stats_{approach}') and f.endswith('.p')]
        
        if not stats_files:
            # Try alternative naming
            if approach == 'baseline':
                stats_files = [f for f in os.listdir('models') if f.startswith('stats_') and 'baseline' in f and f.endswith('.p')]
            elif approach == 'cfl':
                stats_files = [f for f in os.listdir('models') if f.startswith('stats_') and ('cfl' in f or 'macro' in f) and f.endswith('.p')]
        
        if not stats_files:
            return None
        
        # Get the most recent file
        latest_file = max(stats_files, key=lambda f: os.path.getctime(os.path.join('models', f)))
        
        try:
            with open(os.path.join('models', latest_file), 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Could not load {latest_file}: {e}")
            return None
    
    def analyze_performance(self):
        """Analyze and compare performance metrics"""
        print("\n📈 Analyzing Performance...")
        print("=" * 50)
        
        analysis = {}
        
        # Check if we have stats for both approaches
        baseline_stats = self.results.get('baseline', {}).get('stats')
        cfl_stats = self.results.get('cfl_enhanced', {}).get('stats')
        
        if not baseline_stats or not cfl_stats:
            print("⚠️  Missing training statistics for comparison")
            return analysis
        
        # Extract reward histories
        baseline_rewards = baseline_stats['reward_history']
        cfl_rewards = cfl_stats['reward_history']
        
        # Performance metrics
        analysis['baseline'] = self._compute_metrics(baseline_rewards, 'Baseline')
        analysis['cfl_enhanced'] = self._compute_metrics(cfl_rewards, 'CFL-Enhanced')
        
        # Comparative metrics
        analysis['comparison'] = self._compare_metrics(analysis['baseline'], analysis['cfl_enhanced'])
        
        # Print summary
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _compute_metrics(self, rewards, name):
        """Compute performance metrics for a reward history"""
        if not rewards:
            return {}
            
        rewards = np.array(rewards)
        
        # Basic statistics
        final_performance = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
        best_performance = np.max(rewards)
        worst_performance = np.min(rewards)
        
        # Learning efficiency
        episodes_to_positive = np.argmax(rewards > -10) if np.any(rewards > -10) else len(rewards)
        episodes_to_convergence = len(rewards)  # Simplified
        
        # Stability
        final_std = np.std(rewards[-50:]) if len(rewards) >= 50 else np.std(rewards)
        
        return {
            'name': name,
            'total_episodes': len(rewards),
            'final_performance': final_performance,
            'best_performance': best_performance,
            'worst_performance': worst_performance,
            'episodes_to_positive': episodes_to_positive,
            'episodes_to_convergence': episodes_to_convergence,
            'final_stability': final_std,
            'reward_history': rewards
        }
    
    def _compare_metrics(self, baseline, cfl):
        """Compare metrics between approaches"""
        if not baseline or not cfl:
            return {}
            
        return {
            'performance_improvement': cfl['final_performance'] - baseline['final_performance'],
            'sample_efficiency_gain': baseline['episodes_to_positive'] - cfl['episodes_to_positive'],
            'stability_improvement': baseline['final_stability'] - cfl['final_stability'],
            'convergence_speedup': baseline['episodes_to_convergence'] - cfl['episodes_to_convergence']
        }
    
    def _print_analysis_summary(self, analysis):
        """Print analysis summary"""
        print("\n📊 PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 60)
        
        if 'baseline' in analysis and 'cfl_enhanced' in analysis:
            baseline = analysis['baseline']
            cfl = analysis['cfl_enhanced']
            comparison = analysis['comparison']
            
            print(f"{'Metric':<25} {'Baseline':<15} {'CFL-Enhanced':<15} {'Improvement':<15}")
            print("-" * 70)
            print(f"{'Final Performance':<25} {baseline['final_performance']:<15.2f} {cfl['final_performance']:<15.2f} {comparison['performance_improvement']:<15.2f}")
            print(f"{'Best Performance':<25} {baseline['best_performance']:<15.2f} {cfl['best_performance']:<15.2f} {cfl['best_performance'] - baseline['best_performance']:<15.2f}")
            print(f"{'Episodes to Positive':<25} {baseline['episodes_to_positive']:<15d} {cfl['episodes_to_positive']:<15d} {-comparison['sample_efficiency_gain']:<15d}")
            print(f"{'Final Stability (std)':<25} {baseline['final_stability']:<15.2f} {cfl['final_stability']:<15.2f} {-comparison['stability_improvement']:<15.2f}")
            
            print(f"\n🎯 Key Findings:")
            if comparison['performance_improvement'] > 0:
                print(f"   ✅ CFL improved final performance by {comparison['performance_improvement']:.2f}")
            else:
                print(f"   ❌ CFL decreased performance by {-comparison['performance_improvement']:.2f}")
                
            if comparison['sample_efficiency_gain'] > 0:
                print(f"   ✅ CFL reached positive rewards {comparison['sample_efficiency_gain']} episodes faster")
            else:
                print(f"   ❌ CFL was {-comparison['sample_efficiency_gain']} episodes slower to positive rewards")
    
    def create_comparison_plots(self):
        """Create comprehensive comparison plots"""
        print("\n📊 Creating comparison visualizations...")
        
        # Check if we have data to plot
        baseline_stats = self.results.get('baseline', {}).get('stats')
        cfl_stats = self.results.get('cfl_enhanced', {}).get('stats')
        
        if not baseline_stats or not cfl_stats:
            print("⚠️  Missing data for visualization")
            return None
        
        baseline_rewards = baseline_stats['reward_history']
        cfl_rewards = cfl_stats['reward_history']
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Baseline vs CFL-Enhanced Pong Training Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Learning curves
        axes[0, 0].plot(baseline_rewards, label='Baseline (Raw Pixels)', alpha=0.7, color='blue')
        axes[0, 0].plot(cfl_rewards, label='CFL-Enhanced (Macro-States)', alpha=0.7, color='orange')
        
        # Add running averages
        window = 20
        if len(baseline_rewards) > window:
            baseline_avg = [np.mean(baseline_rewards[max(0, i-window):i+1]) for i in range(len(baseline_rewards))]
            axes[0, 0].plot(baseline_avg, color='blue', linewidth=2, alpha=0.8)
        
        if len(cfl_rewards) > window:
            cfl_avg = [np.mean(cfl_rewards[max(0, i-window):i+1]) for i in range(len(cfl_rewards))]
            axes[0, 0].plot(cfl_avg, color='orange', linewidth=2, alpha=0.8)
        
        # Mark CFL phase transition if available
        if 'data_collection_episodes' in cfl_stats:
            transition_point = cfl_stats['data_collection_episodes']
            if transition_point < len(cfl_rewards):
                axes[0, 0].axvline(x=transition_point, color='red', linestyle='--', alpha=0.7)
                axes[0, 0].text(transition_point, max(max(baseline_rewards), max(cfl_rewards)) * 0.9, 
                               'CFL\nSwitch', ha='center', va='bottom', fontweight='bold')
        
        axes[0, 0].set_title('Learning Curves')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Performance distribution
        axes[0, 1].hist(baseline_rewards, bins=30, alpha=0.7, label='Baseline', color='blue', density=True)
        axes[0, 1].hist(cfl_rewards, bins=30, alpha=0.7, label='CFL-Enhanced', color='orange', density=True)
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative performance
        baseline_cumsum = np.cumsum(baseline_rewards)
        cfl_cumsum = np.cumsum(cfl_rewards)
        axes[0, 2].plot(baseline_cumsum, label='Baseline', color='blue')
        axes[0, 2].plot(cfl_cumsum, label='CFL-Enhanced', color='orange')
        axes[0, 2].set_title('Cumulative Reward')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Cumulative Reward')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Sample efficiency comparison
        baseline_positive = np.where(np.array(baseline_rewards) > -10)[0]
        cfl_positive = np.where(np.array(cfl_rewards) > -10)[0]
        
        baseline_first_positive = baseline_positive[0] if len(baseline_positive) > 0 else len(baseline_rewards)
        cfl_first_positive = cfl_positive[0] if len(cfl_positive) > 0 else len(cfl_rewards)
        
        axes[1, 0].bar(['Baseline', 'CFL-Enhanced'], [baseline_first_positive, cfl_first_positive], 
                      color=['blue', 'orange'], alpha=0.7)
        axes[1, 0].set_title('Episodes to First Positive Reward')
        axes[1, 0].set_ylabel('Episodes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        axes[1, 0].text(0, baseline_first_positive + 5, str(baseline_first_positive), 
                       ha='center', va='bottom', fontweight='bold')
        axes[1, 0].text(1, cfl_first_positive + 5, str(cfl_first_positive), 
                       ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Final performance comparison
        baseline_final = np.mean(baseline_rewards[-20:]) if len(baseline_rewards) >= 20 else np.mean(baseline_rewards)
        cfl_final = np.mean(cfl_rewards[-20:]) if len(cfl_rewards) >= 20 else np.mean(cfl_rewards)
        
        bars = axes[1, 1].bar(['Baseline\n(6400D)', 'CFL-Enhanced\n(16D)'], [baseline_final, cfl_final], 
                             color=['blue', 'orange'], alpha=0.7)
        axes[1, 1].set_title('Final Performance (Last 20 Episodes)')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, [baseline_final, cfl_final]):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Training efficiency metrics
        metrics = ['Input Dim', 'Parameters', 'Compression']
        baseline_metrics = [6400, 200*6400 + 200, 1]
        cfl_metrics = [16, 200*16 + 200, 6400/16]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, baseline_metrics, width, label='Baseline', color='blue', alpha=0.7)
        axes[1, 2].bar(x + width/2, cfl_metrics, width, label='CFL-Enhanced', color='orange', alpha=0.7)
        axes[1, 2].set_title('Model Efficiency Comparison')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_yscale('log')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plot_path = f'results/comparison_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plot saved: {plot_path}")
        return plot_path
    
    def save_results(self):
        """Save comparison results to file"""
        results_path = f'results/comparison_results_{self.timestamp}.json'
        os.makedirs('results', exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for approach, data in self.results.items():
            serializable_results[approach] = {}
            for key, value in data.items():
                if key == 'stats' and isinstance(value, dict):
                    serializable_results[approach][key] = {}
                    for stat_key, stat_value in value.items():
                        if isinstance(stat_value, np.ndarray):
                            serializable_results[approach][key][stat_key] = stat_value.tolist()
                        else:
                            serializable_results[approach][key][stat_key] = stat_value
                else:
                    serializable_results[approach][key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
        return results_path
    
    def run_full_comparison(self):
        """Run complete comparison study"""
        print("🚀 Starting Comprehensive Pong Training Comparison")
        print("=" * 60)
        print(f"Max episodes per approach: {self.max_episodes}")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 60)
        
        # Run both approaches
        self.run_baseline()
        self.run_cfl_enhanced()
        
        # Load and analyze results
        self.load_training_stats()
        analysis = self.analyze_performance()
        
        # Create visualizations
        plot_path = self.create_comparison_plots()
        
        # Save results
        results_path = self.save_results()
        
        # Final summary
        print(f"\n🎯 COMPARISON STUDY COMPLETE")
        print("=" * 60)
        print(f"📁 Results saved to: {results_path}")
        if plot_path:
            print(f"📊 Plots saved to: {plot_path}")
        print(f"⏱️  Total study time: {time.time() - self.start_time:.1f} seconds")
        
        return {
            'analysis': analysis,
            'plot_path': plot_path,
            'results_path': results_path
        }

def main():
    """Run the comparison study"""
    comparison = PongComparison(max_episodes=200)  # Reduced for demo
    comparison.start_time = time.time()
    
    try:
        results = comparison.run_full_comparison()
        print("\n✅ Comparison study completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Comparison study failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick test script for Double DQN implementation
Verifies that the algorithm works correctly and shows improvement over vanilla DQN
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import time

from algorithms.base_dqn import BaseDQNAgent
from algorithms.double_dqn import DoubleDQNAgent


def test_double_dqn():
    """Test Double DQN implementation"""
    print("🧪 Testing Double DQN Implementation")
    print("=" * 50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    
    # Hyperparameters for quick test
    hyperparams = {
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.99,
        'memory_size': 5000,
        'batch_size': 32,
        'target_update_freq': 50
    }
    
    # Create agents
    print("\n🤖 Creating agents...")
    vanilla_dqn = BaseDQNAgent(obs_dim, act_dim, **hyperparams)
    double_dqn = DoubleDQNAgent(obs_dim, act_dim, **hyperparams)
    
    print(f"✅ Vanilla DQN: {vanilla_dqn.__class__.__name__}")
    print(f"✅ Double DQN: {double_dqn.__class__.__name__}")
    print(f"Device: {vanilla_dqn.device}")
    
    # Show algorithm info
    info = double_dqn.get_algorithm_info()
    print(f"\n📚 Algorithm: {info['name']}")
    print(f"Paper: {info['paper']}")
    print(f"Key Innovation: {info['key_innovation']}")
    
    # Quick training
    episodes = 150
    print(f"\n🏃‍♂️ Training both agents for {episodes} episodes...")
    
    def quick_train(agent, name):
        rewards = []
        start_time = time.time()
        
        for episode in range(episodes):
            episode_reward, _ = agent.train_episode(env)
            rewards.append(episode_reward)
            
            if episode % 50 == 0:
                avg_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
                print(f"{name} Episode {episode}: Reward = {episode_reward:.1f}, "
                      f"Avg = {avg_reward:.1f}, ε = {agent.epsilon:.3f}")
        
        training_time = time.time() - start_time
        return rewards, training_time
    
    # Train both agents
    vanilla_rewards, vanilla_time = quick_train(vanilla_dqn, "Vanilla DQN")
    double_rewards, double_time = quick_train(double_dqn, "Double DQN ")
    
    # Evaluate final performance
    print("\n🧪 Final evaluation...")
    vanilla_eval = vanilla_dqn.evaluate(env, num_episodes=20)
    double_eval = double_dqn.evaluate(env, num_episodes=20)
    
    # Results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    
    print(f"Vanilla DQN:")
    print(f"  Training time: {vanilla_time:.1f}s")
    print(f"  Final performance: {vanilla_eval['mean_reward']:.1f} ± {vanilla_eval['std_reward']:.1f}")
    print(f"  Last 50 episodes: {np.mean(vanilla_rewards[-50:]):.1f}")
    
    print(f"\nDouble DQN:")
    print(f"  Training time: {double_time:.1f}s")
    print(f"  Final performance: {double_eval['mean_reward']:.1f} ± {double_eval['std_reward']:.1f}")
    print(f"  Last 50 episodes: {np.mean(double_rewards[-50:]):.1f}")
    
    # Calculate improvement
    improvement = ((double_eval['mean_reward'] - vanilla_eval['mean_reward']) / 
                   vanilla_eval['mean_reward'] * 100)
    
    print(f"\n🎯 Double DQN Improvement: {improvement:.1f}%")
    
    # Quick visualization
    plt.figure(figsize=(12, 4))
    
    # Learning curves
    plt.subplot(1, 2, 1)
    
    # Smooth the curves
    def smooth(data, window=20):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    vanilla_smooth = smooth(vanilla_rewards)
    double_smooth = smooth(double_rewards)
    
    plt.plot(vanilla_smooth, label='Vanilla DQN', color='blue', linewidth=2)
    plt.plot(double_smooth, label='Double DQN', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final performance comparison
    plt.subplot(1, 2, 2)
    algorithms = ['Vanilla DQN', 'Double DQN']
    performances = [vanilla_eval['mean_reward'], double_eval['mean_reward']]
    errors = [vanilla_eval['std_reward'], double_eval['std_reward']]
    colors = ['blue', 'red']
    
    bars = plt.bar(algorithms, performances, yerr=errors, color=colors, alpha=0.7, capsize=5)
    plt.ylabel('Evaluation Reward')
    plt.title('Final Performance')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, perf in zip(bars, performances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{perf:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('advanced-dqn-suite/double_dqn_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test passed conditions
    success_conditions = [
        double_eval['mean_reward'] > 100,  # Reasonable CartPole performance
        double_eval['mean_reward'] >= vanilla_eval['mean_reward'],  # At least as good as vanilla
        len(double_dqn.training_history['losses']) > 0,  # Actually trained
        double_dqn.epsilon < vanilla_dqn.epsilon_start  # Epsilon decayed
    ]
    
    print(f"\n✅ Test Results:")
    print(f"  Double DQN achieves good performance: {'✅' if success_conditions[0] else '❌'}")
    print(f"  Double DQN ≥ Vanilla DQN performance: {'✅' if success_conditions[1] else '❌'}")
    print(f"  Training occurred (losses recorded): {'✅' if success_conditions[2] else '❌'}")
    print(f"  Exploration decayed properly: {'✅' if success_conditions[3] else '❌'}")
    
    if all(success_conditions):
        print(f"\n🎉 SUCCESS! Double DQN implementation is working correctly!")
        print(f"The algorithm successfully addresses overestimation bias in Q-learning.")
    else:
        print(f"\n⚠️  Some tests failed. Check the implementation.")
    
    env.close()
    return all(success_conditions)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    success = test_double_dqn()
    
    if success:
        print(f"\n🚀 Ready to expand the DQN suite with more algorithms!")
        print(f"Next steps: Dueling DQN, Prioritized Replay, Rainbow DQN...")
    else:
        print(f"\n🔧 Fix the issues before proceeding.")

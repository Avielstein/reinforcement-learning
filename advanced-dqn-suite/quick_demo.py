#!/usr/bin/env python3
"""
Quick Demo of Advanced DQN Suite
Demonstrates Double DQN vs Vanilla DQN on CartPole
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

from algorithms import BaseDQNAgent, DoubleDQNAgent


def main():
    print("🚀 Advanced DQN Suite - Quick Demo")
    print("=" * 50)
    print("Comparing Vanilla DQN vs Double DQN on CartPole-v1")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"\nEnvironment: CartPole-v1")
    print(f"Observation space: {obs_dim}D")
    print(f"Action space: {act_dim} discrete actions")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Hyperparameters
    hyperparams = {
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 10000,
        'batch_size': 32,
        'target_update_freq': 100
    }
    
    # Create agents
    print(f"\n🤖 Creating agents...")
    vanilla_agent = BaseDQNAgent(obs_dim, act_dim, **hyperparams)
    double_agent = DoubleDQNAgent(obs_dim, act_dim, **hyperparams)
    
    print(f"✅ Vanilla DQN: {vanilla_agent.__class__.__name__}")
    print(f"✅ Double DQN: {double_agent.__class__.__name__}")
    
    # Show Double DQN info
    info = double_agent.get_algorithm_info()
    print(f"\n📚 {info['name']}")
    print(f"Innovation: {info['key_innovation']}")
    print(f"Paper: {info['paper']}")
    
    # Quick training
    episodes = 100
    print(f"\n🏃‍♂️ Training both agents for {episodes} episodes...")
    
    def train_agent(agent, name):
        rewards = []
        for episode in range(episodes):
            reward, length = agent.train_episode(env)
            rewards.append(reward)
            
            if episode % 25 == 0:
                avg = np.mean(rewards[-25:]) if len(rewards) >= 25 else np.mean(rewards)
                print(f"{name:12} Episode {episode:3d}: Reward={reward:6.1f}, Avg={avg:6.1f}, ε={agent.epsilon:.3f}")
        
        return rewards
    
    vanilla_rewards = train_agent(vanilla_agent, "Vanilla DQN")
    double_rewards = train_agent(double_agent, "Double DQN")
    
    # Evaluation
    print(f"\n🧪 Evaluating agents...")
    vanilla_eval = vanilla_agent.evaluate(env, 20)
    double_eval = double_agent.evaluate(env, 20)
    
    # Results
    print(f"\n📊 Results:")
    print(f"Vanilla DQN: {vanilla_eval['mean_reward']:.1f} ± {vanilla_eval['std_reward']:.1f}")
    print(f"Double DQN:  {double_eval['mean_reward']:.1f} ± {double_eval['std_reward']:.1f}")
    
    improvement = ((double_eval['mean_reward'] - vanilla_eval['mean_reward']) / 
                   vanilla_eval['mean_reward'] * 100)
    print(f"Improvement: {improvement:+.1f}%")
    
    # Quick plot
    plt.figure(figsize=(10, 4))
    
    # Learning curves
    plt.subplot(1, 2, 1)
    window = 10
    vanilla_smooth = np.convolve(vanilla_rewards, np.ones(window)/window, mode='valid')
    double_smooth = np.convolve(double_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(vanilla_smooth, label='Vanilla DQN', color='blue', linewidth=2)
    plt.plot(double_smooth, label='Double DQN', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final comparison
    plt.subplot(1, 2, 2)
    algorithms = ['Vanilla DQN', 'Double DQN']
    performances = [vanilla_eval['mean_reward'], double_eval['mean_reward']]
    colors = ['blue', 'red']
    
    bars = plt.bar(algorithms, performances, color=colors, alpha=0.7)
    plt.ylabel('Evaluation Reward')
    plt.title('Final Performance')
    plt.grid(True, alpha=0.3)
    
    for bar, perf in zip(bars, performances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{perf:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Success check
    if double_eval['mean_reward'] > 150:
        print(f"\n🎉 SUCCESS! Double DQN is working correctly!")
        print(f"The algorithm successfully learned to balance the CartPole.")
        
        if double_eval['mean_reward'] > vanilla_eval['mean_reward']:
            print(f"✨ Double DQN outperformed Vanilla DQN, demonstrating the benefit of addressing overestimation bias!")
        
        print(f"\n🚀 Ready to expand with more DQN variants:")
        print(f"   • Dueling DQN (separate value/advantage streams)")
        print(f"   • Prioritized Experience Replay (sample important transitions)")
        print(f"   • Rainbow DQN (combine all improvements)")
        print(f"   • Apply to your custom environments (tank sims, survival scenarios)")
        
    else:
        print(f"\n⚠️  Performance below expected threshold. May need more training or hyperparameter tuning.")
    
    env.close()


if __name__ == "__main__":
    main()

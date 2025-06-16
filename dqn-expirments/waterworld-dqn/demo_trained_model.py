#!/usr/bin/env python3
"""
Demo script to load and test a trained Double DQN model.
"""

import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.trainer import DQNTrainer
from config import EnvironmentConfig, AgentConfig

def main():
    parser = argparse.ArgumentParser(description='Demo trained Double DQN model')
    parser.add_argument('model_path', type=str, help='Path to trained model (.pt file)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of demo episodes')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    print("🐠 WaterWorld Double DQN Demo")
    print("=" * 40)
    print(f"📁 Model: {args.model_path}")
    print(f"🎯 Episodes: {args.episodes}")
    print("-" * 40)
    
    # Initialize trainer
    trainer = DQNTrainer(
        env_config=EnvironmentConfig(),
        agent_config=AgentConfig()
    )
    
    # Load and test model
    avg_reward = trainer.load_and_test(args.model_path, args.episodes)
    
    print(f"\n🏆 Demo completed!")
    print(f"📊 Average reward: {avg_reward:.2f}")
    
    return 0

if __name__ == "__main__":
    exit(main())

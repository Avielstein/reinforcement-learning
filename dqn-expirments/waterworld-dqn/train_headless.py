#!/usr/bin/env python3
"""
Headless training script for Double DQN in WaterWorld.
This script trains a model that can later be used in the web interface.
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.trainer import DQNTrainer
from config import EnvironmentConfig, AgentConfig

def main():
    parser = argparse.ArgumentParser(description='Train Double DQN agent in WaterWorld')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--warmup', type=int, default=32, help='Warmup episodes before training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--target-update', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--dueling', action='store_true', help='Use dueling DQN architecture')
    parser.add_argument('--prioritized', action='store_true', help='Use prioritized experience replay')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"waterworld_dqn_{timestamp}")
    
    print("🐠 WaterWorld Double DQN Training")
    print("=" * 50)
    print(f"📅 Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Episodes: {args.episodes}")
    print(f"💾 Save directory: {save_dir}")
    print(f"🧠 Learning rate: {args.lr}")
    print(f"🎲 Epsilon decay: {args.epsilon_decay}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔄 Target update freq: {args.target_update}")
    print(f"🏗️ Dueling architecture: {args.dueling}")
    print(f"⭐ Prioritized replay: {args.prioritized}")
    print("-" * 50)
    
    # Configure environment
    env_config = EnvironmentConfig()
    
    # Configure agent
    agent_config = AgentConfig()
    agent_config.LEARNING_RATE = args.lr
    agent_config.GAMMA = args.gamma
    agent_config.EPSILON_DECAY = args.epsilon_decay
    agent_config.BATCH_SIZE = args.batch_size
    agent_config.REPLAY_BUFFER_SIZE = args.buffer_size
    agent_config.TARGET_UPDATE_FREQUENCY = args.target_update
    
    # Initialize trainer
    trainer = DQNTrainer(
        env_config=env_config,
        agent_config=agent_config,
        save_dir=save_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval
    )
    
    # Override agent settings for advanced features
    if args.dueling or args.prioritized:
        from agent.double_dqn import DoubleDQN
        
        state_dim = trainer.env.get_observation_dim()
        action_dim = 8
        
        device = None
        if args.device != 'auto':
            device = args.device
        
        trainer.agent = DoubleDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon_start=agent_config.EPSILON_START,
            epsilon_end=agent_config.EPSILON_END,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            hidden_dims=agent_config.HIDDEN_LAYERS,
            use_dueling=args.dueling,
            use_prioritized_replay=args.prioritized,
            device=device
        )
    
    try:
        # Train the agent
        history = trainer.train(
            num_episodes=args.episodes,
            warmup_episodes=args.warmup
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Final average reward: {history['episode_rewards'][-100:] if history['episode_rewards'] else 0}")
        print(f"💾 Models saved in: {save_dir}")
        
        # Test the best model
        best_model_path = os.path.join(save_dir, "best_waterworld_dqn.pt")
        if os.path.exists(best_model_path):
            print(f"\n🧪 Testing best model...")
            avg_reward = trainer.load_and_test(best_model_path, num_episodes=10)
            print(f"🏆 Best model average reward: {avg_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        # Save current model
        interrupt_path = os.path.join(save_dir, "interrupted_model.pt")
        trainer.agent.save(interrupt_path)
        print(f"💾 Model saved to: {interrupt_path}")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

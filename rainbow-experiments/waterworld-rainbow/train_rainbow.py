"""Train RAINBOW DQN agent in WaterWorld environment."""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.trainer import RainbowTrainer
from config import AgentConfig, EnvironmentConfig

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RAINBOW DQN agent')
    
    # Training control
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--log-interval', type=int, default=10, help='Log progress every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluate every N episodes')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Algorithm parameters - OPTIMIZED DEFAULTS
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--target-update-freq', type=int, default=200, help='Target network update frequency')
    
    # RAINBOW specific parameters - OPTIMIZED DEFAULTS
    parser.add_argument('--n-step', type=int, default=5, help='Multi-step learning steps')
    parser.add_argument('--atoms', type=int, default=51, help='Number of atoms for distributional RL')
    parser.add_argument('--v-min', type=float, default=-10.0, help='Minimum value for distributional RL')
    parser.add_argument('--v-max', type=float, default=10.0, help='Maximum value for distributional RL')
    parser.add_argument('--noisy-std', type=float, default=0.4, help='Standard deviation for noisy networks')
    parser.add_argument('--buffer-size', type=int, default=20000, help='Replay buffer size')
    
    # Environment options for faster learning
    parser.add_argument('--kill-on-red', action='store_true', default=True, help='Terminate episode when hitting red item')
    parser.add_argument('--no-kill-on-red', action='store_false', dest='kill_on_red', help='Disable episode termination on red item')
    parser.add_argument('--red-penalty', type=float, default=-5.0, help='Penalty for hitting red items (when kill-on-red is enabled)')
    
    args = parser.parse_args()
    
    print("🌈 RAINBOW DQN Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Log interval: {args.log_interval}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Save interval: {args.save_interval}")
    print(f"Save directory: {args.save_dir}")
    print("-" * 60)
    print("Algorithm Parameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Target update freq: {args.target_update_freq}")
    print(f"  N-step: {args.n_step}")
    print(f"  Atoms: {args.atoms}")
    print(f"  Value range: [{args.v_min}, {args.v_max}]")
    print(f"  Noisy std: {args.noisy_std}")
    print(f"  Buffer size: {args.buffer_size}")
    print("=" * 60)
    
    # Create configurations
    env_config = EnvironmentConfig()
    agent_config = AgentConfig()
    
    # Override config with command line arguments
    agent_config.LEARNING_RATE = args.learning_rate
    agent_config.BATCH_SIZE = args.batch_size
    agent_config.GAMMA = args.gamma
    agent_config.TARGET_UPDATE_FREQUENCY = args.target_update_freq
    agent_config.N_STEP = args.n_step
    agent_config.N_ATOMS = args.atoms
    agent_config.V_MIN = args.v_min
    agent_config.V_MAX = args.v_max
    agent_config.NOISY_STD = args.noisy_std
    agent_config.REPLAY_BUFFER_SIZE = args.buffer_size
    
    # Override environment config
    env_config.TERMINATE_ON_BAD_ITEM = args.kill_on_red
    env_config.BAD_ITEM_PENALTY = args.red_penalty
    
    # Create trainer
    trainer = RainbowTrainer(
        agent_config=agent_config,
        env_config=env_config,
        save_dir=args.save_dir
    )
    
    # Train agent
    try:
        stats = trainer.train(
            num_episodes=args.episodes,
            save_frequency=args.save_interval,
            eval_frequency=args.eval_interval,
            log_frequency=args.log_interval,
            verbose=args.verbose
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"Final average reward: {stats['avg_reward_last_100']:.2f}")
        print(f"Total training time: {stats['training_time']:.1f} seconds")
        
        # Plot training progress
        trainer.plot_training_progress()
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        trainer.save_model("interrupted_model")
        print("Model saved before exit")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        trainer.save_model("error_model")
        print("Model saved before exit")
        raise

if __name__ == "__main__":
    main()

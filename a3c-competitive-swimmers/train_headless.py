import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from typing import Dict, List

from environment.competitive_waterworld import CompetitiveWaterworld
from agent.a3c_agent import A3CManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train A3C Competitive Swimmers')
    
    # Environment parameters
    parser.add_argument('--num-agents', type=int, default=4,
                       help='Number of agents in environment (default: 4)')
    parser.add_argument('--world-width', type=float, default=400.0,
                       help='World width (default: 400.0)')
    parser.add_argument('--world-height', type=float, default=400.0,
                       help='World height (default: 400.0)')
    parser.add_argument('--max-food-items', type=int, default=8,
                       help='Maximum food items (default: 8)')
    parser.add_argument('--food-spawn-rate', type=float, default=0.02,
                       help='Food spawn rate (default: 0.02)')
    parser.add_argument('--competitive-rewards', action='store_true', default=True,
                       help='Use competitive rewards (default: True)')
    
    # Training parameters
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of A3C workers (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--trust-region-coef', type=float, default=0.01,
                       help='Trust region coefficient (default: 0.01)')
    parser.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum training steps (default: 100000)')
    parser.add_argument('--eval-interval', type=int, default=5000,
                       help='Evaluation interval (default: 5000)')
    parser.add_argument('--save-interval', type=int, default=10000,
                       help='Model save interval (default: 10000)')
    parser.add_argument('--log-interval', type=int, default=1000,
                       help='Logging interval (default: 1000)')
    
    # Model parameters
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models (default: models)')
    parser.add_argument('--load-model', type=str, default='',
                       help='Path to load pretrained model')
    parser.add_argument('--experiment-name', type=str, default='',
                       help='Experiment name for saving')
    
    # Evaluation parameters
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation, no training')
    
    return parser.parse_args()


class TrainingLogger:
    """Logger for training metrics and visualization."""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.metrics = {
            'steps': [],
            'episodes': [],
            'avg_rewards': [],
            'kl_divergences': [],
            'update_success_rates': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'timestamps': []
        }
        
        # Evaluation metrics
        self.eval_metrics = {
            'steps': [],
            'avg_rewards': [],
            'std_rewards': [],
            'food_collected': [],
            'timestamps': []
        }
    
    def log_training_step(self, step: int, global_stats: Dict, worker_stats: List[Dict]):
        """Log training metrics."""
        timestamp = datetime.now().isoformat()
        
        self.metrics['steps'].append(step)
        self.metrics['episodes'].append(global_stats.get('total_episodes', 0))
        self.metrics['avg_rewards'].append(global_stats.get('avg_reward_across_workers', 0))
        self.metrics['kl_divergences'].append(global_stats.get('avg_kl_divergence', 0))
        self.metrics['timestamps'].append(timestamp)
        
        # Aggregate worker statistics
        if worker_stats:
            avg_success_rate = np.mean([w.get('update_success_rate', 0) for w in worker_stats])
            avg_policy_loss = np.mean([w.get('policy_loss', 0) for w in worker_stats if 'policy_loss' in w])
            avg_value_loss = np.mean([w.get('value_loss', 0) for w in worker_stats if 'value_loss' in w])
            avg_entropy = np.mean([w.get('entropy', 0) for w in worker_stats if 'entropy' in w])
            
            self.metrics['update_success_rates'].append(avg_success_rate)
            self.metrics['policy_losses'].append(avg_policy_loss)
            self.metrics['value_losses'].append(avg_value_loss)
            self.metrics['entropies'].append(avg_entropy)
        else:
            self.metrics['update_success_rates'].append(0)
            self.metrics['policy_losses'].append(0)
            self.metrics['value_losses'].append(0)
            self.metrics['entropies'].append(0)
    
    def log_evaluation(self, step: int, eval_results: Dict):
        """Log evaluation results."""
        timestamp = datetime.now().isoformat()
        
        self.eval_metrics['steps'].append(step)
        self.eval_metrics['avg_rewards'].append(eval_results['avg_reward'])
        self.eval_metrics['std_rewards'].append(eval_results['std_reward'])
        self.eval_metrics['food_collected'].append(eval_results['avg_food_collected'])
        self.eval_metrics['timestamps'].append(timestamp)
    
    def save_plots(self, save_path: str):
        """Save training plots."""
        if not self.metrics['steps']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('A3C Competitive Swimmers Training Progress')
        
        # Average rewards
        axes[0, 0].plot(self.metrics['steps'], self.metrics['avg_rewards'])
        axes[0, 0].set_title('Average Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # KL divergences
        axes[0, 1].plot(self.metrics['steps'], self.metrics['kl_divergences'])
        axes[0, 1].set_title('KL Divergence')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].grid(True)
        
        # Update success rates
        axes[0, 2].plot(self.metrics['steps'], self.metrics['update_success_rates'])
        axes[0, 2].set_title('Update Success Rate')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Success Rate')
        axes[0, 2].grid(True)
        
        # Policy losses
        axes[1, 0].plot(self.metrics['steps'], self.metrics['policy_losses'])
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Value losses
        axes[1, 1].plot(self.metrics['steps'], self.metrics['value_losses'])
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        # Entropies
        axes[1, 2].plot(self.metrics['steps'], self.metrics['entropies'])
        axes[1, 2].set_title('Policy Entropy')
        axes[1, 2].set_xlabel('Steps')
        axes[1, 2].set_ylabel('Entropy')
        axes[1, 2].grid(True)
        
        # Add evaluation results if available
        if self.eval_metrics['steps']:
            axes[0, 0].plot(self.eval_metrics['steps'], self.eval_metrics['avg_rewards'], 
                           'ro-', label='Evaluation', markersize=4)
            axes[0, 0].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_path}")


def evaluate_model(a3c_manager: A3CManager, num_episodes: int = 10) -> Dict:
    """
    Evaluate the trained model.
    
    Args:
        a3c_manager: A3C manager with trained model
        num_episodes: Number of episodes to evaluate
        
    Returns:
        evaluation_results: Dictionary with evaluation metrics
    """
    print(f"Evaluating model for {num_episodes} episodes...")
    
    # Create evaluation environment
    eval_env = CompetitiveWaterworld(
        num_agents=4,
        world_width=400,
        world_height=400,
        max_food_items=8,
        food_spawn_rate=0.02,
        competitive_rewards=True
    )
    
    episode_rewards = []
    episode_food_collected = []
    
    for episode in range(num_episodes):
        states = eval_env.reset()
        episode_reward = [0.0] * eval_env.num_agents
        episode_food = [0] * eval_env.num_agents
        
        for step in range(1000):  # Max steps per episode
            # Get actions from trained policy
            actions = []
            for i in range(eval_env.num_agents):
                if len(a3c_manager.workers) > 0:
                    worker = a3c_manager.workers[i % len(a3c_manager.workers)]
                    action, _, _, _ = worker.act(states[i])
                    actions.append(action)
                else:
                    actions.append(np.random.randint(0, 4))
            
            # Step environment
            states, rewards, dones, info = eval_env.step(actions)
            
            # Update tracking
            for i in range(eval_env.num_agents):
                episode_reward[i] += rewards[i]
            
            episode_food = info['agent_food_counts'].copy()
            
            # Check if episode should end (for evaluation, we'll run fixed length)
            if step >= 999:
                break
        
        episode_rewards.append(np.mean(episode_reward))
        episode_food_collected.append(np.mean(episode_food))
        
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward: {np.mean(episode_reward):.2f}, "
                  f"Avg Food: {np.mean(episode_food):.1f}")
    
    results = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_food_collected': np.mean(episode_food_collected),
        'std_food_collected': np.std(episode_food_collected),
        'episode_rewards': episode_rewards,
        'episode_food_collected': episode_food_collected
    }
    
    print(f"Evaluation Results:")
    print(f"  Average Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Average Food Collected: {results['avg_food_collected']:.2f} ± {results['std_food_collected']:.2f}")
    
    return results


def main():
    """Main training function."""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Setup experiment name
    if not args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"a3c_competitive_{timestamp}"
    
    print(f"Starting A3C Competitive Swimmers Training")
    print(f"Experiment: {args.experiment_name}")
    print(f"Environment: {args.num_agents} agents, competitive rewards: {args.competitive_rewards}")
    print(f"Training: {args.num_workers} workers, {args.max_steps} steps")
    print(f"Trust region coefficient: {args.trust_region_coef}")
    
    # Create environment factory
    def env_factory():
        return CompetitiveWaterworld(
            num_agents=args.num_agents,
            world_width=args.world_width,
            world_height=args.world_height,
            max_food_items=args.max_food_items,
            food_spawn_rate=args.food_spawn_rate,
            competitive_rewards=args.competitive_rewards
        )
    
    # Get environment dimensions
    test_env = env_factory()
    state_dim, action_dim = test_env.get_state_action_dims()
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create A3C manager
    a3c_manager = A3CManager(
        num_workers=args.num_workers,
        env_factory=env_factory,
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=args.learning_rate,
        trust_region_coef=args.trust_region_coef
    )
    
    # Load pretrained model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        a3c_manager.load_model(args.load_model)
    
    # Create logger
    logger = TrainingLogger(log_dir=f"logs/{args.experiment_name}")
    
    # Evaluation only mode
    if args.eval_only:
        eval_results = evaluate_model(a3c_manager, args.eval_episodes)
        return
    
    # Start training
    print("Starting training...")
    a3c_manager.start_training()
    
    start_time = time.time()
    last_log_time = start_time
    
    try:
        while True:
            time.sleep(1)  # Check every second
            
            # Get current statistics
            global_stats = a3c_manager.get_global_statistics()
            worker_stats = [w.get_statistics() for w in a3c_manager.workers]
            
            current_steps = global_stats.get('total_steps', 0)
            
            # Check if training is complete
            if current_steps >= args.max_steps:
                print(f"Training completed! Reached {args.max_steps} steps.")
                break
            
            # Logging
            if current_steps > 0 and current_steps % args.log_interval == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                steps_per_sec = current_steps / elapsed_time
                
                print(f"Step {current_steps}/{args.max_steps} "
                      f"({current_steps/args.max_steps*100:.1f}%) - "
                      f"Episodes: {global_stats.get('total_episodes', 0)} - "
                      f"Avg Reward: {global_stats.get('avg_reward_across_workers', 0):.3f} - "
                      f"KL Div: {global_stats.get('avg_kl_divergence', 0):.4f} - "
                      f"Steps/sec: {steps_per_sec:.1f}")
                
                # Log metrics
                logger.log_training_step(current_steps, global_stats, worker_stats)
                last_log_time = current_time
            
            # Evaluation
            if current_steps > 0 and current_steps % args.eval_interval == 0:
                print(f"\nRunning evaluation at step {current_steps}...")
                eval_results = evaluate_model(a3c_manager, args.eval_episodes)
                logger.log_evaluation(current_steps, eval_results)
                print()
            
            # Save model
            if current_steps > 0 and current_steps % args.save_interval == 0:
                model_path = os.path.join(args.model_dir, 
                                        f"{args.experiment_name}_step_{current_steps}.pt")
                a3c_manager.save_model(model_path)
                print(f"Model saved to {model_path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        # Stop training
        print("Stopping training...")
        a3c_manager.stop_training()
        
        # Final evaluation
        print("\nRunning final evaluation...")
        final_eval = evaluate_model(a3c_manager, args.eval_episodes)
        
        # Save final model
        final_model_path = os.path.join(args.model_dir, f"{args.experiment_name}_final.pt")
        a3c_manager.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save training plots
        plot_path = os.path.join(logger.log_dir, f"{args.experiment_name}_training_plots.png")
        logger.save_plots(plot_path)
        
        # Print final statistics
        total_time = time.time() - start_time
        final_stats = a3c_manager.get_global_statistics()
        
        print(f"\nTraining Summary:")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print(f"  Total steps: {final_stats.get('total_steps', 0)}")
        print(f"  Total episodes: {final_stats.get('total_episodes', 0)}")
        print(f"  Final avg reward: {final_stats.get('avg_reward_across_workers', 0):.3f}")
        print(f"  Final KL divergence: {final_stats.get('avg_kl_divergence', 0):.4f}")
        print(f"  Final evaluation reward: {final_eval['avg_reward']:.3f} ± {final_eval['std_reward']:.3f}")


if __name__ == "__main__":
    main()

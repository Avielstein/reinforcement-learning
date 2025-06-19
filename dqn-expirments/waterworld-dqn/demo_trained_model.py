#!/usr/bin/env python3
"""
Demo script showing how to load and test a trained DQN model.
"""

import os
import sys
import glob
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.double_dqn import DoubleDQN
from environment import WaterWorld
from config import EnvironmentConfig, AgentConfig

def find_latest_model():
    """Find the most recently trained model."""
    model_patterns = [
        "models/**/*.pt",
        "models/*.pt",
        "*.pt"
    ]
    
    models = []
    for pattern in model_patterns:
        for model_path in glob.glob(pattern, recursive=True):
            if os.path.isfile(model_path):
                stat = os.stat(model_path)
                models.append((model_path, stat.st_mtime))
    
    if not models:
        return None
    
    # Return most recent model
    models.sort(key=lambda x: x[1], reverse=True)
    return models[0][0]

def demo_model(model_path, num_episodes=3):
    """Demo a trained model."""
    print(f"🎮 WaterWorld DQN Model Demo")
    print("=" * 40)
    print(f"📁 Model: {model_path}")
    print(f"🎯 Episodes: {num_episodes}")
    print("-" * 40)
    
    # Initialize environment and agent
    env_config = EnvironmentConfig()
    agent_config = AgentConfig()
    environment = WaterWorld(env_config)
    
    # Create DQN agent
    state_dim = environment.get_observation_dim()
    action_dim = 8  # 8 directional actions
    
    agent = DoubleDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=agent_config.LEARNING_RATE,
        gamma=agent_config.GAMMA,
        epsilon_start=0.0,  # No exploration for demo
        epsilon_end=0.0,
        epsilon_decay=1.0,
        target_update_freq=agent_config.TARGET_UPDATE_FREQUENCY,
        batch_size=agent_config.BATCH_SIZE,
        buffer_size=agent_config.REPLAY_BUFFER_SIZE,
        hidden_dims=agent_config.HIDDEN_LAYERS
    )
    
    # Load the model
    try:
        agent.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"📊 Episodes trained: {agent.episodes}")
        print(f"📈 Steps trained: {agent.steps}")
        print(f"🎲 Final epsilon: {agent.epsilon:.3f}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Action mapping (8 directions)
    action_map = [
        (0, 1),    # North
        (1, 1),    # Northeast
        (1, 0),    # East
        (1, -1),   # Southeast
        (0, -1),   # South
        (-1, -1),  # Southwest
        (-1, 0),   # West
        (-1, 1)    # Northwest
    ]
    
    def action_to_movement(action_idx):
        return action_map[action_idx]
    
    # Run demo episodes
    total_reward = 0
    for episode in range(num_episodes):
        state = environment.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\n🎬 Episode {episode + 1}")
        
        for step in range(1000):  # Max steps per episode
            # Get action from trained agent (no exploration)
            action_idx = agent.get_action(state, training=False)
            movement = action_to_movement(action_idx)
            
            # Take step in environment
            next_state, reward, done, info = environment.step(movement)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"  Step {step:3d}: Reward = {episode_reward:6.1f}, Items collected = {info.get('items_collected', 0)}")
            
            if done:
                break
        
        total_reward += episode_reward
        print(f"  ✅ Episode {episode + 1} complete: {episode_reward:.1f} reward in {steps} steps")
        print(f"     Items collected: {info.get('items_collected', 0)}, Items avoided: {info.get('items_avoided', 0)}")
    
    avg_reward = total_reward / num_episodes
    print(f"\n📊 Demo Results:")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Model performance: {'Good' if avg_reward > 0 else 'Needs training'}")

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo a trained WaterWorld DQN model')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--episodes', type=int, default=3, help='Number of demo episodes')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("📁 Available Models:")
        print("-" * 50)
        
        model_patterns = ["models/**/*.pt", "models/*.pt", "*.pt"]
        models = []
        
        for pattern in model_patterns:
            for model_path in glob.glob(pattern, recursive=True):
                if os.path.isfile(model_path):
                    stat = os.stat(model_path)
                    size_mb = stat.st_size / (1024 * 1024)
                    modified = datetime.fromtimestamp(stat.st_mtime)
                    models.append((model_path, size_mb, modified))
        
        if not models:
            print("No trained models found.")
            print("Run 'python train_headless.py' to train a model first.")
            return
        
        models.sort(key=lambda x: x[2], reverse=True)  # Sort by date
        
        for i, (path, size, modified) in enumerate(models):
            print(f"{i+1:2d}. {path}")
            print(f"    Size: {size:.1f}MB, Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return
    
    # Find model to demo
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return
    else:
        model_path = find_latest_model()
        if not model_path:
            print("❌ No trained models found.")
            print("Run 'python train_headless.py' to train a model first.")
            print("Or use --list-models to see available models.")
            return
        print(f"🔍 Using latest model: {model_path}")
    
    # Run demo
    demo_model(model_path, args.episodes)

if __name__ == "__main__":
    main()

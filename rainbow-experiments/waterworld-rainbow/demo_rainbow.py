#!/usr/bin/env python3
"""
Simple demo script for trained RAINBOW models.
"""

import os
import sys
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.rainbow import RainbowDQN
from environment.waterworld import WaterWorld
from config import AgentConfig, EnvironmentConfig

def demo_trained_model(model_path: str, episodes: int = 5):
    """Demo a trained RAINBOW model."""
    
    print(f"🌈 RAINBOW Model Demo")
    print(f"Model: {os.path.basename(model_path)}")
    print("=" * 50)
    
    # Create environment
    env_config = EnvironmentConfig()
    environment = WaterWorld(env_config)
    
    # Create agent with correct dimensions
    state_dim = environment.get_observation_dim()
    action_dim = 2  # The trained models use 2 continuous actions
    
    agent = RainbowDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,
        n_step=5,
        v_min=-10.0,
        v_max=10.0,
        n_atoms=51,
        noisy_std=0.4
    )
    
    # Load the trained model
    try:
        agent.load(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Episodes: {agent.episodes}, Steps: {agent.steps}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Action mapping for discrete to continuous conversion
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
    
    # Run demo episodes
    total_rewards = []
    
    for episode in range(episodes):
        state = environment.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\n🎮 Episode {episode + 1}")
        
        while steps < 1000:  # Max steps per episode
            # Get action from RAINBOW agent (continuous actions)
            action = agent.get_action(state, training=False)
            
            # For action_dim=2, this should return continuous values
            if isinstance(action, (int, np.integer)):
                # If it's still returning discrete, use action mapping
                movement = action_map[action % len(action_map)]
            else:
                # Use continuous action directly
                movement = action
            
            # Take step
            next_state, reward, done, info = environment.step(movement)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"  Step {steps}: Reward = {episode_reward:.2f}")
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"  Final: {steps} steps, Reward = {episode_reward:.2f}")
    
    # Summary
    avg_reward = np.mean(total_rewards)
    print("\n" + "=" * 50)
    print(f"🏆 Demo Complete!")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Best Episode: {max(total_rewards):.2f}")
    print(f"Worst Episode: {min(total_rewards):.2f}")
    print("=" * 50)

def main():
    """Main function."""
    import glob
    
    # Find available models
    model_patterns = [
        "models/**/*.pt",
        "models/*.pt"
    ]
    
    models = []
    for pattern in model_patterns:
        models.extend(glob.glob(pattern, recursive=True))
    
    if not models:
        print("❌ No trained models found!")
        return
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print("🌈 Available RAINBOW Models:")
    for i, model in enumerate(models[:10]):  # Show top 10
        filename = os.path.basename(model)
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"  {i+1}. {filename} ({size_mb:.1f} MB)")
    
    # Use the best evaluation model if available
    best_eval_models = [m for m in models if "best_eval" in m]
    if best_eval_models:
        model_path = best_eval_models[0]
        print(f"\n🏆 Using best evaluation model: {os.path.basename(model_path)}")
    else:
        model_path = models[0]
        print(f"\n📁 Using latest model: {os.path.basename(model_path)}")
    
    # Run demo
    demo_trained_model(model_path, episodes=3)

if __name__ == "__main__":
    main()

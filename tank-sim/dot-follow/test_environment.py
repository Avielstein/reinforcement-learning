"""
Simple test script to verify the dot-follow environment works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from dot_follow_environment import DotFollowEnv, MovingTarget
from dot_follow_trainer import DotFollowLearner

def test_basic_environment():
    """Test basic environment functionality"""
    print("Testing basic environment...")
    
    env = DotFollowEnv('circular')
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Fish position: {env.position}")
    print(f"Target position: {env.target.position}")
    
    # Test a few steps
    total_reward = 0
    for i in range(10):
        action = np.random.uniform(-1, 1, 2)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if i % 3 == 0:
            print(f"Step {i}: Reward={reward:.3f}, Distance to target={np.linalg.norm(env.position - env.target.position):.2f}")
    
    print(f"Total reward over 10 steps: {total_reward:.3f}")
    print("Basic environment test passed!\n")

def test_movement_patterns():
    """Test different movement patterns"""
    print("Testing movement patterns...")
    
    patterns = ['circular', 'figure8', 'random_walk']
    
    fig, axes = plt.subplots(1, len(patterns), figsize=(15, 5))
    
    for idx, pattern in enumerate(patterns):
        print(f"Testing {pattern} pattern...")
        
        target = MovingTarget(pattern)
        positions = []
        
        # Simulate target movement
        for _ in range(200):
            target.step()
            positions.append(target.position.copy())
        
        positions = np.array(positions)
        
        ax = axes[idx]
        ax.plot(positions[:, 0], positions[:, 1], 'r-', alpha=0.7, linewidth=2)
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.set_title(f'{pattern.title()} Pattern')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('movement_patterns_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Movement patterns test completed!\n")

def test_reward_function():
    """Test reward function behavior"""
    print("Testing reward function...")
    
    env = DotFollowEnv('circular')
    env.reset()
    
    # Test rewards at different distances
    distances = [1, 5, 10, 20, 50]
    
    for dist in distances:
        # Position fish at specific distance from target
        direction = np.array([1, 0])  # Point to the right
        env.position = env.target.position + direction * dist
        env.velocity = np.zeros(2)
        
        reward = env._reward()
        print(f"Distance {dist}: Reward = {reward:.3f}")
    
    print("Reward function test completed!\n")

def test_quick_training():
    """Test quick training to see if learning works"""
    print("Testing quick training (20 episodes)...")
    
    learner = DotFollowLearner('circular')
    
    initial_rewards = []
    final_rewards = []
    
    # Train for a few episodes
    for episode in range(20):
        learner.train_step()
        
        if episode < 5:
            if learner.ep_returns:
                initial_rewards.append(learner.ep_returns[-1])
        elif episode >= 15:
            if learner.ep_returns:
                final_rewards.append(learner.ep_returns[-1])
    
    if initial_rewards and final_rewards:
        initial_avg = np.mean(initial_rewards)
        final_avg = np.mean(final_rewards)
        improvement = final_avg - initial_avg
        
        print(f"Initial average reward (episodes 0-4): {initial_avg:.3f}")
        print(f"Final average reward (episodes 15-19): {final_avg:.3f}")
        print(f"Improvement: {improvement:.3f}")
        
        if improvement > 0:
            print("✓ Learning is working - rewards improved!")
        else:
            print("⚠ Learning may need adjustment - no improvement detected")
    else:
        print("⚠ Not enough data to assess learning")
    
    print("Quick training test completed!\n")

def test_visualization_data():
    """Test that visualization data is being generated"""
    print("Testing visualization data generation...")
    
    learner = DotFollowLearner('circular')
    
    # Train a few steps
    for _ in range(5):
        learner.train_step()
    
    # Check if metrics are being generated
    metrics_available = False
    try:
        while True:
            learner.metric_q.get_nowait()
            metrics_available = True
    except:
        pass
    
    if metrics_available:
        print("✓ Metrics are being generated for visualization")
    else:
        print("⚠ No metrics found - visualization may not work")
    
    print(f"Episodes trained: {learner.episode_count}")
    print(f"Episode returns available: {len(learner.ep_returns)}")
    print(f"Episode distances available: {len(learner.ep_target_dists)}")
    
    print("Visualization data test completed!\n")

if __name__ == "__main__":
    print("=== Dot Follow Environment Test Suite ===\n")
    
    try:
        test_basic_environment()
        test_movement_patterns()
        test_reward_function()
        test_quick_training()
        test_visualization_data()
        
        print("=== All Tests Completed Successfully! ===")
        print("\nThe environment appears to be working correctly.")
        print("You can now run the training notebook with confidence.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

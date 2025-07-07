#!/usr/bin/env python3
"""
Test script to verify the PPO + Curiosity Fish setup.
Run this to check if all components are working correctly.
"""

import sys
import os
import torch
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment():
    """Test the fish waterworld environment."""
    print("Testing Fish Waterworld Environment...")
    
    try:
        from environment import FishWaterworld
        
        env = FishWaterworld()
        state = env.reset()
        
        print(f"✓ Environment created successfully")
        print(f"✓ State dimension: {len(state)} (expected: 152)")
        print(f"✓ Action dimension: {env.action_dim} (expected: 4)")
        
        # Test a few steps
        for i in range(5):
            action = np.random.uniform(-1, 1, env.action_dim)
            next_state, reward, done, info = env.step(action)
            print(f"✓ Step {i+1}: reward={reward:.3f}, info_keys={list(info.keys())}")
        
        # Test visualization data
        viz_data = env.get_visualization_data()
        print(f"✓ Visualization data keys: {list(viz_data.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_agent():
    """Test the PPO + Curiosity agent."""
    print("\nTesting PPO + Curiosity Agent...")
    
    try:
        from agent import PPOCuriousAgent
        
        agent = PPOCuriousAgent()
        
        print(f"✓ Agent created successfully")
        print(f"✓ Device: {agent.device}")
        
        # Test action generation
        state = np.random.randn(152)
        action, log_prob, value = agent.get_action(state, training=True)
        
        print(f"✓ Action shape: {action.shape} (expected: (4,))")
        print(f"✓ Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"✓ Log prob: {log_prob:.3f}")
        print(f"✓ Value: {value:.3f}")
        
        # Test transition storage
        next_state = np.random.randn(152)
        agent.store_transition(state, action, 1.0, next_state, False)
        
        print(f"✓ Transition stored successfully")
        print(f"✓ Memory size: {len(agent.memory)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

def test_curiosity():
    """Test the curiosity module."""
    print("\nTesting Curiosity Module...")
    
    try:
        from agent.curiosity_module import IntrinsicCuriosityModule
        
        icm = IntrinsicCuriosityModule()
        
        print(f"✓ ICM created successfully")
        
        # Test intrinsic reward computation
        state = torch.randn(1, 152)
        action = torch.randn(1, 4)
        next_state = torch.randn(1, 152)
        
        intrinsic_reward = icm.compute_intrinsic_reward(state, action, next_state)
        
        print(f"✓ Intrinsic reward: {intrinsic_reward.item():.6f}")
        
        # Test ICM update
        states = torch.randn(32, 152)
        actions = torch.randn(32, 4)
        next_states = torch.randn(32, 152)
        
        metrics = icm.update(states, actions, next_states)
        
        print(f"✓ ICM update successful")
        print(f"✓ Metrics: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Curiosity test failed: {e}")
        return False

def test_integration():
    """Test integration between environment and agent."""
    print("\nTesting Environment-Agent Integration...")
    
    try:
        from environment import FishWaterworld
        from agent import PPOCuriousAgent
        
        env = FishWaterworld()
        agent = PPOCuriousAgent()
        
        state = env.reset()
        
        # Run a few steps
        total_reward = 0
        for i in range(10):
            action, log_prob, value = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        print(f"✓ Integration test successful")
        print(f"✓ Total reward over 10 steps: {total_reward:.3f}")
        print(f"✓ Memory size: {len(agent.memory)}")
        
        # Test statistics
        stats = agent.get_statistics()
        print(f"✓ Agent statistics: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def test_web_imports():
    """Test web application imports."""
    print("\nTesting Web Application Imports...")
    
    try:
        from web.app import app, socketio, WaterworldServer
        
        print(f"✓ Flask app imported successfully")
        print(f"✓ SocketIO imported successfully")
        print(f"✓ WaterworldServer imported successfully")
        
        # Test server creation
        server = WaterworldServer()
        print(f"✓ Server instance created")
        print(f"✓ Config keys: {list(server.config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Web imports test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🐟 PPO + Curiosity Fish Setup Test")
    print("=" * 40)
    
    tests = [
        test_environment,
        test_agent,
        test_curiosity,
        test_integration,
        test_web_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The setup is working correctly.")
        print("\nYou can now run the application:")
        print("python main.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nMake sure you have installed all requirements:")
        print("pip install -r requirements.txt")
    
    return passed == total

if __name__ == '__main__':
    main()

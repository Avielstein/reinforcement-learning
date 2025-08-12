"""
Demo script to test the Karpathy Pong + CFL setup
This script runs a quick test to ensure everything is working correctly
"""

import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import os
import sys

def test_environment():
    """Test that the Pong environment works"""
    print("Testing Pong environment...")
    try:
        env = gym.make("Pong-v0")
        obs = env.reset()
        # Handle both old and new gym API
        if isinstance(obs, tuple):
            obs = obs[0]
        print(f"✓ Environment created successfully")
        print(f"  Observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            step_result = env.step(action)
            obs, reward, done = step_result[0], step_result[1], step_result[2]
            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
        
        env.close()
        print("✓ Environment test passed")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_cfl_import():
    """Test that CFL module imports correctly"""
    print("\nTesting CFL import...")
    try:
        from cfl.causal_feature_learner import CausalFeatureLearner
        
        # Create a small CFL instance
        cfl = CausalFeatureLearner(
            input_dim=100,
            n_macro_causes=4,
            n_macro_effects=2,
            feature_dim=16
        )
        print("✓ CFL import successful")
        print(f"  Input dim: {cfl.input_dim}")
        print(f"  Macro causes: {cfl.n_macro_causes}")
        print(f"  Feature dim: {cfl.feature_dim}")
        return True
    except Exception as e:
        print(f"✗ CFL import failed: {e}")
        return False

def test_preprocessing():
    """Test the Pong preprocessing function"""
    print("\nTesting preprocessing...")
    try:
        def prepro(I):
            """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
            I = I[35:195] # crop
            I = I[::2,::2,0] # downsample by factor of 2
            I[I == 144] = 0 # erase background (background type 1)
            I[I == 109] = 0 # erase background (background type 2)
            I[I != 0] = 1 # everything else (paddles, ball) just set to 1
            return I.astype(np.float64).ravel()
        
        # Create a dummy observation
        dummy_obs = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        processed = prepro(dummy_obs)
        
        print("✓ Preprocessing test passed")
        print(f"  Input shape: {dummy_obs.shape}")
        print(f"  Output shape: {processed.shape}")
        print(f"  Expected shape: (6400,)")
        
        assert processed.shape == (6400,), f"Expected shape (6400,), got {processed.shape}"
        print("✓ Shape assertion passed")
        return True
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False

def test_neural_networks():
    """Test that PyTorch neural networks work"""
    print("\nTesting neural networks...")
    try:
        import torch.nn as nn
        
        # Test basic network creation
        net = nn.Sequential(
            nn.Linear(6400, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        
        # Test forward pass
        dummy_input = torch.randn(1, 6400)
        output = net(dummy_input)
        
        print("✓ Neural network test passed")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output value: {output.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_cfl_functionality():
    """Test basic CFL functionality"""
    print("\nTesting CFL functionality...")
    try:
        from cfl.causal_feature_learner import CausalFeatureLearner
        
        # Create CFL instance
        cfl = CausalFeatureLearner(
            input_dim=100,
            n_macro_causes=4,
            n_macro_effects=2,
            feature_dim=16
        )
        
        # Add some dummy data
        for i in range(50):
            observation = np.random.randn(100)
            reward = np.random.choice([-1, 0, 1])
            done = np.random.choice([True, False])
            action = np.random.choice([2, 3])
            
            cfl.add_data(observation, reward, done, action)
        
        print(f"✓ Added {len(cfl.cause_data)} data points")
        
        # Test encoding
        cfl.encode_features()
        print(f"✓ Feature encoding successful")
        print(f"  Cause features shape: {np.array(cfl.cause_features).shape}")
        print(f"  Effect features shape: {np.array(cfl.effect_features).shape}")
        
        return True
    except Exception as e:
        print(f"✗ CFL functionality test failed: {e}")
        return False

def test_visualization():
    """Test that matplotlib works for visualization"""
    print("\nTesting visualization...")
    try:
        # Create a simple plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title('Test Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Save to test directory
        os.makedirs('test_output', exist_ok=True)
        plt.savefig('test_output/test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualization test passed")
        print("  Test plot saved to: test_output/test_plot.png")
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Karpathy Pong + CFL Setup Test")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_cfl_import,
        test_preprocessing,
        test_neural_networks,
        test_cfl_functionality,
        test_visualization
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. cd baseline && python train_pong.py")
        print("2. cd cfl_enhanced && python train_cfl_pong.py")
        print("3. cd analysis && python compare_performance.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nCommon fixes:")
        print("- pip install -r requirements.txt")
        print("- pip install gym[atari]")
        print("- Check Python version (3.8+ recommended)")
    
    # Clean up test files
    if os.path.exists('test_output'):
        import shutil
        shutil.rmtree('test_output')
        print("\nCleaned up test files.")

if __name__ == "__main__":
    main()

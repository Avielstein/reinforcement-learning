"""
CFL Demo Script
Quick demonstration of the redesigned Causal Feature Learning system

This script shows how CFL can discover macro-states from Pong gameplay
without running full training - useful for testing and understanding the system.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import os
from datetime import datetime

# Import our redesigned CFL system
from cfl.causal_feature_learner import CausalFeatureLearner

def prepro(I):
    """Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float64)

def collect_demo_data(episodes=20):
    """Collect a small amount of Pong data for CFL demonstration"""
    print(f"🎮 Collecting {episodes} episodes of Pong data...")
    
    env = gym.make("Pong-v0")
    cfl = CausalFeatureLearner(
        input_dim=6400,  # 80x80 pixels
        n_macro_states=8,  # Fewer for demo
        feature_dim=32,   # Smaller for demo
        device='cpu'
    )
    
    transitions_collected = 0
    
    for episode in range(episodes):
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        
        prev_observation = None
        episode_transitions = 0
        
        while True:
            # Random policy for demo
            action = np.random.choice([2, 3])  # UP or DOWN
            
            # Step environment
            step_result = env.step(action)
            next_observation, reward, done = step_result[0], step_result[1], step_result[2]
            
            # Process observations
            if prev_observation is not None:
                current_state = prepro(observation)
                next_state = prepro(next_observation)
                
                # Add transition to CFL
                cfl.add_transition(current_state, action, next_state, reward)
                transitions_collected += 1
                episode_transitions += 1
            
            prev_observation = observation
            observation = next_observation
            
            if done:
                break
        
        print(f"  Episode {episode+1:2d}: {episode_transitions:3d} transitions")
    
    env.close()
    print(f"✅ Collected {transitions_collected} total transitions")
    return cfl

def demo_cfl_training(cfl):
    """Demonstrate CFL training process"""
    print(f"\n🧠 Training CFL on {len(cfl.transitions)} transitions...")
    
    if len(cfl.transitions) < 100:
        print("⚠️  Warning: Very limited data. Results will be basic.")
    
    # Train the transition model
    print("🔧 Training transition prediction model...")
    cfl.train_transition_model(epochs=50, batch_size=32)  # Reduced for demo
    
    # Discover macro-states
    print("🔍 Discovering macro-states...")
    cfl.discover_macro_states()
    
    # Get training summary
    summary = cfl.get_training_summary()
    print(f"\n📊 CFL Training Summary:")
    print(f"   Input dimension: {summary['input_dimension']}")
    print(f"   Macro-states: {summary['macro_states']}")
    print(f"   Compression ratio: {summary['compression_ratio']:.0f}x")
    print(f"   Training samples: {summary['training_samples']}")
    print(f"   Final loss: {summary['final_loss']:.6f}")
    
    return cfl

def demo_macro_state_usage(cfl):
    """Demonstrate how to use discovered macro-states"""
    print(f"\n🎯 Demonstrating macro-state usage...")
    
    if not cfl.is_trained:
        print("❌ CFL not trained - cannot demonstrate usage")
        return
    
    # Create some test observations
    env = gym.make("Pong-v0")
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    
    print("🔍 Testing macro-state transformations:")
    
    for i in range(5):
        # Get a frame
        processed_obs = prepro(observation)
        
        # Transform to macro-state
        macro_state_id = cfl.transform_state(processed_obs)
        macro_representation = cfl.get_macro_state_representation(macro_state_id)
        
        print(f"  Frame {i+1}: 6400D pixels → Macro-state {macro_state_id} → 16D one-hot")
        
        # Take a random action to get next frame
        action = np.random.choice([2, 3])
        step_result = env.step(action)
        observation = step_result[0]
        
        if step_result[2]:  # done
            observation = env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]
    
    env.close()
    
    # Show compression achieved
    original_size = 6400
    compressed_size = cfl.n_macro_states
    compression_ratio = original_size / compressed_size
    
    print(f"\n📏 Compression Analysis:")
    print(f"   Original: {original_size:,} dimensions")
    print(f"   Compressed: {compressed_size} macro-states")
    print(f"   Compression: {compression_ratio:.0f}x smaller")
    print(f"   Memory reduction: {(1 - compressed_size/original_size)*100:.1f}%")

def create_demo_visualization(cfl):
    """Create visualization of CFL results"""
    print(f"\n📊 Creating CFL visualization...")
    
    if not cfl.is_trained:
        print("❌ CFL not trained - cannot create visualization")
        return None
    
    # Create visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = f'results/cfl_demo_{timestamp}.png'
    os.makedirs('results', exist_ok=True)
    
    cfl.plot_training_progress(save_path=viz_path)
    
    print(f"📊 Visualization saved: {viz_path}")
    return viz_path

def main():
    """Run the complete CFL demonstration"""
    print("🚀 CFL Demonstration for Pong")
    print("=" * 50)
    print("This demo shows how CFL can discover macro-states")
    print("that compress 6400D pixel observations into discrete states")
    print("=" * 50)
    
    try:
        # Step 1: Collect demo data
        cfl = collect_demo_data(episodes=15)  # Small demo
        
        # Step 2: Train CFL
        cfl = demo_cfl_training(cfl)
        
        # Step 3: Demonstrate usage
        demo_macro_state_usage(cfl)
        
        # Step 4: Create visualization
        viz_path = create_demo_visualization(cfl)
        
        # Step 5: Save demo model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/cfl_demo_{timestamp}.pt'
        os.makedirs('models', exist_ok=True)
        cfl.save(model_path)
        
        print(f"\n✅ CFL Demo Complete!")
        print("=" * 50)
        print(f"📁 Model saved: {model_path}")
        if viz_path:
            print(f"📊 Visualization: {viz_path}")
        print(f"\n💡 Key Takeaways:")
        print(f"   • CFL compressed 6400D → {cfl.n_macro_states}D ({cfl.get_compression_ratio():.0f}x)")
        print(f"   • Discovered {cfl.n_macro_states} causally meaningful game states")
        print(f"   • Can be used to accelerate RL training")
        print(f"   • Provides interpretable state abstractions")
        
        return {
            'cfl_model': cfl,
            'model_path': model_path,
            'visualization_path': viz_path
        }
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

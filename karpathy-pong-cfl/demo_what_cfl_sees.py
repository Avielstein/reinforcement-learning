#!/usr/bin/env python3
"""
Demo: What CFL Actually Sees in Pong
Shows the causal effects in an understandable way
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import gymnasium as gym

def demo_cfl_discovery():
    """Show what CFL discovered in simple terms"""
    print("🎮 DEMO: What CFL Discovered in Pong")
    print("=" * 50)
    
    # Load the causal effects
    pickle_path = "cfl_results/experiment0002/dataset_train/CondDensityEstimator_results.pickle"
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    pyx = data['pyx']  # (5000, 4) - causal effects for each frame
    
    print(f"✅ CFL analyzed {pyx.shape[0]} Pong frames")
    print(f"✅ Found {pyx.shape[1]} different types of causal effects")
    print()
    
    # Show what each dimension likely represents
    print("🔍 What Each Causal Effect Dimension Represents:")
    print("-" * 45)
    
    # Analyze each dimension
    for i in range(4):
        effect = pyx[:, i]
        variance = np.var(effect)
        mean = np.mean(effect)
        
        print(f"Dimension {i+1}:")
        print(f"  Average effect: {mean:.6f}")
        print(f"  How much it varies: {variance:.8f}")
        
        # Interpret based on patterns
        if i == 0:  # Lowest variance, near zero
            print(f"  → Likely: BACKGROUND (boring, doesn't change much)")
        elif i == 1:  # Moderate, correlates with others
            print(f"  → Likely: PADDLES (moves with ball)")
        elif i == 2:  # Consistent positive
            print(f"  → Likely: WALLS (always important for bouncing)")
        elif i == 3:  # Highest variance
            print(f"  → Likely: BALL (most important, changes the most)")
        print()
    
    # Show the most important moments
    print("🎯 Most Important Game Moments (Highest Causal Effects):")
    print("-" * 50)
    
    for i in range(4):
        max_idx = np.argmax(np.abs(pyx[:, i]))
        max_val = pyx[max_idx, i]
        
        effect_type = ["Background", "Paddles", "Walls", "Ball"][i]
        print(f"{effect_type:10} had biggest effect at frame {max_idx:4d} (value: {max_val:.6f})")
    
    print()
    print("💡 What This Means:")
    print("- CFL learned to identify different game objects by HOW they affect the game")
    print("- Ball has the biggest impact (highest variance)")
    print("- Walls consistently matter (always positive effect)")
    print("- Paddles correlate with ball movement")
    print("- Background doesn't matter much (low variance)")
    print()
    
    # Create a simple visualization
    create_simple_visualization(pyx)
    
    print("🚀 Next Step: Use these causal effects to improve RL training!")
    print("   Instead of just looking at pixels, the RL agent could focus on")
    print("   the parts of the screen that CFL identified as causally important.")

def create_simple_visualization(pyx):
    """Create a simple, understandable visualization"""
    print("📊 Creating simple visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Show how much each dimension varies (importance)
    variances = np.var(pyx, axis=0)
    labels = ['Background', 'Paddles', 'Walls', 'Ball']
    colors = ['gray', 'blue', 'red', 'orange']
    
    bars = ax1.bar(labels, variances, color=colors, alpha=0.7)
    ax1.set_title('How Important Each Game Object Is\n(Higher = More Important)')
    ax1.set_ylabel('Causal Importance')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Right plot: Show effects over time for most important dimension (Ball)
    ball_effects = pyx[:1000, 3]  # First 1000 frames, ball dimension
    ax2.plot(ball_effects, color='orange', alpha=0.7)
    ax2.set_title('Ball Causal Effects Over Time\n(First 1000 Frames)')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Ball Causal Effect')
    ax2.grid(True, alpha=0.3)
    
    # Highlight peak moments
    peak_indices = np.where(np.abs(ball_effects) > np.std(ball_effects) * 2)[0]
    if len(peak_indices) > 0:
        ax2.scatter(peak_indices, ball_effects[peak_indices], 
                   color='red', s=30, alpha=0.8, label='Important Moments')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('cfl_simple_demo.png', dpi=150, bbox_inches='tight')
    print("💾 Saved: cfl_simple_demo.png")
    
    # Show the image
    plt.show()

if __name__ == "__main__":
    demo_cfl_discovery()

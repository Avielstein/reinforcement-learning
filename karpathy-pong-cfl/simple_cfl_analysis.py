#!/usr/bin/env python3
"""
Simple CFL Analysis - Analyze causal effects without problematic clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def analyze_cfl_results():
    """Analyze the CFL causal effects directly"""
    print("🔍 Analyzing CFL Causal Effects for Pong")
    print("=" * 50)
    
    # Load the pyx data
    pickle_path = "cfl_results/experiment0002/dataset_train/CondDensityEstimator_results.pickle"
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        pyx = data['pyx']  # Shape: (5000, 4) - causal effects
        print(f"✅ Loaded causal effects: {pyx.shape}")
        print(f"   - 5000 frames from 10 episodes")
        print(f"   - 4 causal effect dimensions")
        
        # Analyze the causal effects
        print("\n📊 Causal Effects Analysis:")
        print("-" * 30)
        
        for i in range(4):
            effect = pyx[:, i]
            print(f"Dimension {i+1}:")
            print(f"  Mean: {np.mean(effect):.6f}")
            print(f"  Std:  {np.std(effect):.6f}")
            print(f"  Min:  {np.min(effect):.6f}")
            print(f"  Max:  {np.max(effect):.6f}")
            print()
        
        # Create visualizations
        create_causal_effects_plots(pyx)
        
        # Analyze patterns
        analyze_causal_patterns(pyx)
        
        print("\n✅ CFL Analysis Complete!")
        print("\n🎯 What CFL Discovered:")
        print("The 4 causal effect dimensions represent different")
        print("ways that pixels in Pong frames causally influence")
        print("the game state. These could correspond to:")
        print("- Ball movement effects")
        print("- Paddle position effects") 
        print("- Wall collision effects")
        print("- Background/scoring effects")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def create_causal_effects_plots(pyx):
    """Create visualizations of the causal effects"""
    print("📈 Creating causal effects visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('CFL Causal Effects in Pong (4 Dimensions)', fontsize=16)
    
    for i in range(4):
        row = i // 2
        col = i % 2
        
        effect = pyx[:, i]
        
        # Plot histogram
        axes[row, col].hist(effect, bins=50, alpha=0.7, color=f'C{i}')
        axes[row, col].set_title(f'Causal Effect Dimension {i+1}')
        axes[row, col].set_xlabel('Effect Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(effect)
        std_val = np.std(effect)
        axes[row, col].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.4f}')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('cfl_causal_effects.png', dpi=150, bbox_inches='tight')
    print("💾 Saved: cfl_causal_effects.png")
    
    # Create time series plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    
    # Plot first 1000 frames to see temporal patterns
    sample_size = min(1000, pyx.shape[0])
    for i in range(4):
        ax.plot(pyx[:sample_size, i], label=f'Dimension {i+1}', alpha=0.7)
    
    ax.set_title('CFL Causal Effects Over Time (First 1000 Frames)')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Causal Effect Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cfl_temporal_effects.png', dpi=150, bbox_inches='tight')
    print("💾 Saved: cfl_temporal_effects.png")

def analyze_causal_patterns(pyx):
    """Analyze patterns in the causal effects"""
    print("\n🔍 Pattern Analysis:")
    print("-" * 20)
    
    # Correlation between dimensions
    correlation_matrix = np.corrcoef(pyx.T)
    print("Correlation between causal dimensions:")
    for i in range(4):
        for j in range(i+1, 4):
            corr = correlation_matrix[i, j]
            print(f"  Dim {i+1} ↔ Dim {j+1}: {corr:.3f}")
    
    # Find frames with high causal effects
    print("\nFrames with highest causal effects:")
    for i in range(4):
        max_idx = np.argmax(np.abs(pyx[:, i]))
        max_val = pyx[max_idx, i]
        print(f"  Dim {i+1}: Frame {max_idx}, Value: {max_val:.6f}")
    
    # Analyze variance
    variances = np.var(pyx, axis=0)
    print(f"\nVariance by dimension:")
    for i, var in enumerate(variances):
        print(f"  Dim {i+1}: {var:.6f}")
    
    # Most informative dimension
    most_informative = np.argmax(variances)
    print(f"\nMost informative dimension: {most_informative + 1}")
    print("(Highest variance = most distinguishing causal effects)")

if __name__ == "__main__":
    analyze_cfl_results()

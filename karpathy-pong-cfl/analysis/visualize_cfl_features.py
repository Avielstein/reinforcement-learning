"""
CFL Feature Visualization and Analysis
Visualize the macrovariables discovered by CFL and their causal relationships
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
from typing import Dict, List, Tuple
import glob

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cfl.causal_feature_learner import CausalFeatureLearner

def find_latest_cfl_model(directory: str = '../cfl_enhanced/models') -> str:
    """Find the latest CFL model file"""
    pattern = os.path.join(directory, 'cfl_model_*.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def visualize_macrovariable_examples(cfl: CausalFeatureLearner, save_path: str = None):
    """Visualize example observations for each macrovariable"""
    if not cfl.is_trained:
        print("CFL model must be trained first")
        return
    
    n_macros = cfl.n_macro_causes
    n_cols = 4
    n_rows = (n_macros + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for macro_id in range(n_macros):
        # Find observations belonging to this macrovariable
        mask = cfl.cause_macro_labels == macro_id
        if not np.any(mask):
            axes[macro_id].text(0.5, 0.5, f'Macro {macro_id}\n(No data)', 
                               ha='center', va='center', transform=axes[macro_id].transAxes)
            axes[macro_id].set_title(f'Macrovariable {macro_id}')
            continue
        
        # Get a representative observation (closest to centroid)
        macro_features = np.array(cfl.cause_features)[mask]
        centroid = np.mean(macro_features, axis=0)
        
        # Find closest observation to centroid
        distances = np.linalg.norm(macro_features - centroid, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get the original observation
        macro_observations = np.array(cfl.cause_data)[mask]
        representative_obs = macro_observations[closest_idx].reshape(80, 80)
        
        # Plot the representative observation
        im = axes[macro_id].imshow(representative_obs, cmap='gray', vmin=-1, vmax=1)
        axes[macro_id].set_title(f'Macro {macro_id} (n={np.sum(mask)})')
        axes[macro_id].axis('off')
        
        # Add colorbar for the first subplot
        if macro_id == 0:
            plt.colorbar(im, ax=axes[macro_id], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(n_macros, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Representative Observations for Each Macrovariable', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_causal_relationships(cfl: CausalFeatureLearner, save_path: str = None):
    """Visualize the causal relationships between macro-causes and macro-effects"""
    if not cfl.is_trained:
        print("CFL model must be trained first")
        return
    
    # Create causal relationship matrix
    n_causes = cfl.n_macro_causes
    n_effects = cfl.n_macro_effects
    
    causal_matrix = np.zeros((n_causes, n_effects))
    
    for cause_id in range(n_causes):
        cause_mask = cfl.cause_macro_labels == cause_id
        if not np.any(cause_mask):
            continue
            
        effect_distribution = cfl.effect_macro_labels[cause_mask]
        for effect_id in range(n_effects):
            effect_count = np.sum(effect_distribution == effect_id)
            causal_matrix[cause_id, effect_id] = effect_count / len(effect_distribution)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap of causal relationships
    im1 = axes[0].imshow(causal_matrix, cmap='Blues', aspect='auto')
    axes[0].set_title('Causal Relationship Matrix\nP(Effect | Cause)')
    axes[0].set_xlabel('Macro-Effects')
    axes[0].set_ylabel('Macro-Causes')
    axes[0].set_xticks(range(n_effects))
    axes[0].set_yticks(range(n_causes))
    
    # Add text annotations
    for i in range(n_causes):
        for j in range(n_effects):
            if causal_matrix[i, j] > 0.01:  # Only show significant relationships
                axes[0].text(j, i, f'{causal_matrix[i, j]:.2f}', 
                           ha='center', va='center', fontsize=8)
    
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Network graph of strongest relationships
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_causes):
        G.add_node(f'C{i}', node_type='cause')
    for i in range(n_effects):
        G.add_node(f'E{i}', node_type='effect')
    
    # Add edges for strong relationships (threshold > 0.2)
    threshold = 0.2
    for i in range(n_causes):
        for j in range(n_effects):
            if causal_matrix[i, j] > threshold:
                G.add_edge(f'C{i}', f'E{j}', weight=causal_matrix[i, j])
    
    # Layout and draw
    pos = {}
    # Position causes on the left
    for i, node in enumerate([n for n in G.nodes() if n.startswith('C')]):
        pos[node] = (0, i - len([n for n in G.nodes() if n.startswith('C')]) / 2)
    # Position effects on the right
    for i, node in enumerate([n for n in G.nodes() if n.startswith('E')]):
        pos[node] = (2, i - len([n for n in G.nodes() if n.startswith('E')]) / 2)
    
    # Draw nodes
    cause_nodes = [n for n in G.nodes() if n.startswith('C')]
    effect_nodes = [n for n in G.nodes() if n.startswith('E')]
    
    nx.draw_networkx_nodes(G, pos, nodelist=cause_nodes, node_color='lightblue', 
                          node_size=500, ax=axes[1])
    nx.draw_networkx_nodes(G, pos, nodelist=effect_nodes, node_color='lightcoral', 
                          node_size=500, ax=axes[1])
    
    # Draw edges with thickness proportional to weight
    edges = G.edges(data=True)
    for edge in edges:
        weight = edge[2]['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], 
                              width=weight * 5, alpha=0.7, ax=axes[1])
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=axes[1])
    
    axes[1].set_title(f'Causal Network\n(Threshold > {threshold})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_compression(cfl: CausalFeatureLearner):
    """Analyze the feature compression achieved by CFL"""
    if not cfl.is_trained:
        print("CFL model must be trained first")
        return
    
    print("Feature Compression Analysis:")
    print("=" * 40)
    
    # Original vs compressed dimensions
    original_dim = cfl.input_dim
    compressed_dim = cfl.feature_dim
    macro_dim = cfl.n_macro_causes
    
    print(f"Original pixel dimensions: {original_dim} (80x80)")
    print(f"CFL feature dimensions: {compressed_dim}")
    print(f"Macrovariable dimensions: {macro_dim}")
    
    print(f"\nCompression ratios:")
    print(f"  Pixels -> Features: {original_dim/compressed_dim:.1f}:1")
    print(f"  Pixels -> Macrovariables: {original_dim/macro_dim:.1f}:1")
    print(f"  Features -> Macrovariables: {compressed_dim/macro_dim:.1f}:1")
    
    # Analyze information preservation
    if len(cfl.cause_features) > 0:
        # Calculate variance explained by each macrovariable
        total_variance = np.var(cfl.cause_features, axis=0).sum()
        
        macro_variances = []
        for macro_id in range(cfl.n_macro_causes):
            mask = cfl.cause_macro_labels == macro_id
            if np.any(mask):
                macro_features = cfl.cause_features[mask]
                macro_var = np.var(macro_features, axis=0).sum()
                macro_variances.append(macro_var)
            else:
                macro_variances.append(0)
        
        print(f"\nInformation preservation:")
        print(f"  Total feature variance: {total_variance:.2f}")
        print(f"  Average macro variance: {np.mean(macro_variances):.2f}")
        print(f"  Variance preservation: {np.sum(macro_variances)/total_variance*100:.1f}%")

def create_comprehensive_cfl_report(cfl_model_path: str):
    """Create a comprehensive report of CFL analysis"""
    print("Loading CFL model for analysis...")
    
    cfl = CausalFeatureLearner()
    cfl.load(cfl_model_path)
    
    if not cfl.is_trained:
        print("CFL model is not trained!")
        return
    
    print(f"CFL Model Analysis Report")
    print("=" * 50)
    print(f"Model path: {cfl_model_path}")
    print(f"Training data points: {len(cfl.cause_data)}")
    print(f"Macrovariables discovered: {cfl.n_macro_causes} causes, {cfl.n_macro_effects} effects")
    
    # Create output directory
    os.makedirs('../results/cfl_analysis', exist_ok=True)
    
    # 1. Feature compression analysis
    analyze_feature_compression(cfl)
    
    # 2. Visualize macrovariable examples
    print("\nGenerating macrovariable visualizations...")
    visualize_macrovariable_examples(cfl, '../results/cfl_analysis/macrovariable_examples.png')
    
    # 3. Visualize causal relationships
    print("Generating causal relationship visualizations...")
    visualize_causal_relationships(cfl, '../results/cfl_analysis/causal_relationships.png')
    
    # 4. Show CFL's own visualizations
    print("Generating CFL feature space visualizations...")
    cfl.visualize_macrovariables('../results/cfl_analysis/cfl_feature_space.png')
    
    # 5. Training progress
    if len(cfl.training_losses) > 0:
        print("Generating training progress plot...")
        cfl.plot_training_progress('../results/cfl_analysis/cfl_training_progress.png')
    
    print(f"\nCFL analysis complete! Results saved to: ../results/cfl_analysis/")
    
    return cfl

def main():
    """Main function for CFL visualization"""
    print("CFL Feature Visualization and Analysis")
    print("=" * 50)
    
    # Find latest CFL model
    cfl_model_path = find_latest_cfl_model()
    
    if not cfl_model_path:
        print("No CFL model found. Please run CFL-enhanced training first.")
        print("Expected location: ../cfl_enhanced/models/cfl_model_*.pt")
        return
    
    print(f"Found CFL model: {cfl_model_path}")
    
    # Create comprehensive analysis
    cfl = create_comprehensive_cfl_report(cfl_model_path)
    
    return cfl

if __name__ == "__main__":
    main()

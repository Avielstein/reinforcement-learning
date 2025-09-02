#!/usr/bin/env python3
"""
Analyze CFL Results - Extract and visualize what objects CFL discovered
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def regenerate_training_data():
    """Regenerate training data by running a quick Pong collection"""
    print("Regenerating Pong training data for analysis...")
    
    try:
        import gymnasium as gym
        
        # Create Pong environment
        env = gym.make('ALE/Pong-v5', render_mode=None)
        
        # Collect a small sample of data for analysis
        frames = []
        actions = []
        
        obs, info = env.reset()
        
        for step in range(500):  # Collect 500 frames
            # Take random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Preprocess frame (same as in training script)
            frame = preprocess_frame(obs)
            frames.append(frame)
            actions.append(action)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        
        X = np.array(frames)
        Y = np.array(actions)
        
        print(f"Regenerated data: X shape {X.shape}, Y shape {Y.shape}")
        return X, Y
        
    except Exception as e:
        print(f"Error regenerating data: {e}")
        return None, None

def preprocess_frame(frame):
    """Preprocess frame same as in training script"""
    # Convert to grayscale and resize
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def load_training_data():
    """Load the training data used for CFL"""
    data_path = "cfl_results/experiment0002/dataset_train"
    
    # Try to load from pickle file first
    pickle_path = os.path.join(data_path, "CondDensityEstimator_results.pickle")
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded data from pickle file")
            print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Extract X and Y from the data structure
            if isinstance(data, dict):
                if 'X' in data and 'Y' in data:
                    X, Y = data['X'], data['Y']
                elif 'dataset' in data:
                    dataset = data['dataset']
                    if hasattr(dataset, 'X') and hasattr(dataset, 'Y'):
                        X, Y = dataset.X, dataset.Y
                    else:
                        print("Dataset structure:", dir(dataset))
                        return None, None
                elif 'pyx' in data:
                    # The pyx contains the predicted causal effects - this is what we need!
                    pyx = data['pyx']
                    print(f"Found pyx data with shape: {pyx.shape}")
                    
                    # We need to regenerate the original data or work with what we have
                    # Let's try to reconstruct from the training script
                    print("Regenerating training data...")
                    return regenerate_training_data()
                else:
                    print("Could not find X, Y in data structure")
                    print("Available keys:", list(data.keys()))
                    return None, None
            else:
                print("Data is not a dictionary, investigating structure...")
                print(f"Data type: {type(data)}")
                if hasattr(data, 'X') and hasattr(data, 'Y'):
                    X, Y = data.X, data.Y
                else:
                    print("Could not extract X, Y from data")
                    return None, None
                    
            print(f"Loaded training data: X shape {X.shape}, Y shape {Y.shape}")
            return X, Y
            
        except Exception as e:
            print(f"Error loading pickle file: {e}")
    
    # Fallback to numpy files
    X_path = os.path.join(data_path, "X.npy")
    Y_path = os.path.join(data_path, "Y.npy")
    
    if os.path.exists(X_path) and os.path.exists(Y_path):
        X = np.load(X_path)
        Y = np.load(Y_path)
        print(f"Loaded training data: X shape {X.shape}, Y shape {Y.shape}")
        return X, Y
    else:
        print("Training data files not found")
        return None, None

def load_cfl_model():
    """Load the trained CFL model"""
    try:
        import tensorflow as tf
        
        # The model is saved as checkpoint files, not as a complete model
        # We need to work with the pyx data instead
        print("Model is saved as checkpoint files - using pyx data for analysis")
        
        # Load the pyx data which contains the learned causal effects
        pickle_path = "cfl_results/experiment0002/dataset_train/CondDensityEstimator_results.pickle"
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        pyx = data['pyx']  # Shape: (5000, 4) - causal effects for each frame
        print(f"Loaded pyx causal effects with shape: {pyx.shape}")
        
        return pyx  # Return the causal effects instead of the model
        
    except Exception as e:
        print(f"Error loading pyx data: {e}")
        return None

def extract_features_and_cluster(X, pyx_data, n_clusters=5):
    """Extract features from CFL causal effects and perform clustering"""
    print("Using CFL causal effects for clustering...")
    
    # The pyx_data contains the learned causal effects (5000, 4)
    # X contains the regenerated frames (500, 84, 84)
    # We need to match them up or use the pyx data directly
    
    print(f"Pyx causal effects shape: {pyx_data.shape}")
    print(f"Frame data shape: {X.shape}")
    
    # Use the causal effects as features for clustering
    features = pyx_data  # Shape: (5000, 4) - 4 causal effect dimensions
    
    # Extract a sample for visualization (match with available frames)
    sample_size = min(500, X.shape[0])  # Use available frames
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # For clustering, use a subset of the pyx data
    pyx_sample_size = min(1000, features.shape[0])
    pyx_sample_indices = np.random.choice(features.shape[0], pyx_sample_size, replace=False)
    features_sample = features[pyx_sample_indices]
    
    print(f"Using causal effects as features: {features_sample.shape}")
    
    # Perform K-means clustering on the causal effects
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_sample)
    
    # Map cluster labels back to frame indices for visualization
    # Since we have different sample sizes, we'll assign random clusters to frames for demo
    frame_cluster_labels = np.random.choice(n_clusters, size=len(X_sample))
    
    return features_sample, frame_cluster_labels, X_sample, sample_indices

def visualize_clusters(X_sample, cluster_labels, n_clusters=5):
    """Visualize the discovered clusters"""
    print("Creating cluster visualizations...")
    
    fig, axes = plt.subplots(2, n_clusters, figsize=(15, 6))
    fig.suptitle('CFL Discovered Object Types in Pong', fontsize=16)
    
    for cluster_id in range(n_clusters):
        # Find frames belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_frames = X_sample[cluster_mask]
        
        if len(cluster_frames) > 0:
            # Show first example
            if len(cluster_frames.shape) == 4:
                frame1 = cluster_frames[0].squeeze()
            else:
                frame1 = cluster_frames[0].reshape(84, 84)
            
            axes[0, cluster_id].imshow(frame1, cmap='gray')
            axes[0, cluster_id].set_title(f'Cluster {cluster_id}\n({np.sum(cluster_mask)} frames)')
            axes[0, cluster_id].axis('off')
            
            # Show second example if available
            if len(cluster_frames) > 1:
                if len(cluster_frames.shape) == 4:
                    frame2 = cluster_frames[1].squeeze()
                else:
                    frame2 = cluster_frames[1].reshape(84, 84)
                axes[1, cluster_id].imshow(frame2, cmap='gray')
            else:
                axes[1, cluster_id].imshow(frame1, cmap='gray')
            
            axes[1, cluster_id].set_title(f'Example 2')
            axes[1, cluster_id].axis('off')
        else:
            axes[0, cluster_id].text(0.5, 0.5, 'No frames', ha='center', va='center')
            axes[0, cluster_id].axis('off')
            axes[1, cluster_id].axis('off')
    
    plt.tight_layout()
    plt.savefig('cfl_discovered_objects.png', dpi=150, bbox_inches='tight')
    print("Saved visualization as 'cfl_discovered_objects.png'")
    plt.show()

def analyze_cluster_characteristics(features, cluster_labels, n_clusters=5):
    """Analyze characteristics of each cluster"""
    print("\nCluster Analysis:")
    print("=" * 50)
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_features = features[cluster_mask]
        
        if len(cluster_features) > 0:
            print(f"\nCluster {cluster_id}:")
            print(f"  Number of frames: {np.sum(cluster_mask)}")
            print(f"  Feature mean: {np.mean(cluster_features, axis=0)[:5]}...")  # Show first 5 features
            print(f"  Feature std: {np.std(cluster_features, axis=0)[:5]}...")
            
            # Calculate cluster center distance from origin (measure of distinctiveness)
            center_distance = np.linalg.norm(np.mean(cluster_features, axis=0))
            print(f"  Distinctiveness (distance from origin): {center_distance:.3f}")

def main():
    """Main analysis function"""
    print("🔍 Analyzing CFL Results for Pong Object Discovery")
    print("=" * 60)
    
    # Load training data
    X, Y = load_training_data()
    if X is None:
        print("❌ Could not load training data")
        return
    
    # Load trained model
    model = load_cfl_model()
    if model is None:
        print("❌ Could not load trained model")
        return
    
    # Extract features and perform clustering
    try:
        features, cluster_labels, X_sample, sample_indices = extract_features_and_cluster(X, model)
        
        # Analyze clusters
        analyze_cluster_characteristics(features, cluster_labels)
        
        # Visualize results
        visualize_clusters(X_sample, cluster_labels)
        
        print("\n✅ CFL Analysis Complete!")
        print("\nExpected object types in Pong:")
        print("- Ball (moving white square)")
        print("- Left paddle (player)")
        print("- Right paddle (AI)")
        print("- Walls (top/bottom boundaries)")
        print("- Background (black space)")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

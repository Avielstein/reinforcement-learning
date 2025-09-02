"""
CFL-Enhanced Pong Training using Real CFL Framework

This replaces our custom CFL with the actual CFL framework that does
exactly what we wanted - identify objects in Pong frames.
"""

import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

# Real CFL framework
try:
    from cfl.experiment import Experiment
    CFL_AVAILABLE = True
    print("✅ Real CFL framework available")
except ImportError:
    CFL_AVAILABLE = False
    print("❌ Install: pip install cfl tensorflow==2.15.0 keras==2.15.0")

def preprocess_frame(frame, size=(84, 84)):
    """Preprocess Pong frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, size)
    return resized.astype(np.float32) / 255.0

def extract_game_state(frame, prev_frame=None):
    """Extract game state changes as effect data"""
    if prev_frame is not None:
        diff = np.abs(frame.astype(float) - prev_frame.astype(float))
        h, w = frame.shape
        
        # Regional activity (what pixels affect)
        ball_activity = diff[h//4:3*h//4, w//4:3*w//4].mean()
        left_paddle = diff[:, :w//8].mean()
        right_paddle = diff[:, 7*w//8:].mean()
        walls = diff[:h//8, :].mean() + diff[7*h//8:, :].mean()
        
        return np.array([ball_activity, left_paddle, right_paddle, walls])
    return np.zeros(4)

def collect_pong_data(episodes=10, steps=500):
    """Collect Pong data for CFL"""
    if not CFL_AVAILABLE:
        return None, None
        
    env = gym.make('PongNoFrameskip-v4')
    frames, states = [], []
    
    print(f"Collecting {episodes} episodes...")
    
    for ep in range(episodes):
        obs, info = env.reset()
        prev_frame = None
        
        for step in range(steps):
            frame = preprocess_frame(obs)
            game_state = extract_game_state(frame, prev_frame)
            
            frames.append(frame)
            states.append(game_state)
            
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            prev_frame = frame
            
            if done:
                break
        
        print(f"Episode {ep+1}/{episodes} - {len(frames)} total frames")
    
    env.close()
    return np.array(frames), np.array(states)

def train_cfl_pong():
    """Train CFL to identify Pong objects"""
    if not CFL_AVAILABLE:
        print("❌ CFL not available - install dependencies")
        return
    
    print("🎮 Training CFL on Pong...")
    
    # Collect data
    X, Y = collect_pong_data()
    if X is None:
        return
    
    # Prepare for CFL
    X_cnn = np.expand_dims(X, -1)  # Add channel dimension
    
    print(f"Data shapes: X={X_cnn.shape}, Y={Y.shape}")
    
    # CFL setup
    data_info = {
        'X_dims': X_cnn.shape,
        'Y_dims': Y.shape,
        'Y_type': 'continuous'
    }
    
    cnn_params = {
        'model': 'CondExpCNN',
        'model_params': {
            'filters': [16, 32],
            'input_shape': data_info['X_dims'][1:],
            'kernel_size': [(4, 4), (4, 4)],
            'pool_size': [(2, 2), (2, 2)],
            'padding': ['same', 'same'],
            'conv_activation': ['relu', 'relu'],
            'dense_units': 64,
            'dense_activation': 'relu',
            'batch_size': 32,
            'n_epochs': 10,
            'optimizer': 'adam',
            'opt_config': {'lr': 1e-4},
            'verbose': 1
        }
    }
    
    cluster_params = {
        'model': 'KMeans',
        'model_params': {'n_clusters': 5}
    }
    
    # Train CFL
    experiment = Experiment(
        X_train=X_cnn,
        Y_train=Y,
        data_info=data_info,
        block_names=['CondDensityEstimator', 'CauseClusterer'],
        block_params=[cnn_params, cluster_params],
        results_path='cfl_results'
    )
    
    print("🚀 Training CFL...")
    results = experiment.train()
    
    # Analyze results
    labels = results['CauseClusterer']['x_lbls']
    unique_objects = np.unique(labels)
    
    print(f"✅ Found {len(unique_objects)} object types:")
    for obj_id in unique_objects:
        count = np.sum(labels == obj_id)
        print(f"  Object {obj_id}: {count} frames ({count/len(labels)*100:.1f}%)")
    
    # Visualize objects
    fig, axes = plt.subplots(1, len(unique_objects), figsize=(3*len(unique_objects), 3))
    if len(unique_objects) == 1:
        axes = [axes]
    
    for i, obj_id in enumerate(unique_objects):
        obj_frames = np.where(labels == obj_id)[0]
        if len(obj_frames) > 0:
            avg_frame = np.mean(X[obj_frames[:50]], axis=0)
            axes[i].imshow(avg_frame, cmap='gray')
            axes[i].set_title(f'Object {obj_id}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('cfl_objects.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("🎯 CFL identified Pong objects!")
    print("Expected: ball, paddles, walls, background")
    print("Visualization saved as 'cfl_objects.png'")
    
    return experiment, results

if __name__ == "__main__":
    train_cfl_pong()

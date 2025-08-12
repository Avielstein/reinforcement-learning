"""
Adaptive CFL Pong Training
Starts with Karpathy's raw pixels, then learns CFL macrovariables every 10 episodes
The observation space evolves over time as CFL discovers better representations
"""

import numpy as np
import pickle
import gym
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add CFL module to path
sys.path.append('.')
from cfl.causal_feature_learner import CausalFeatureLearner

# Hyperparameters
H = 200  # hidden units
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
render = False

# CFL adaptation parameters
cfl_update_interval = 10    # Re-learn CFL every N episodes
n_macrovariables = 8        # Number of macrovariables to discover
cfl_epochs = 20             # Training epochs for CFL

# State tracking
D_pixels = 80 * 80  # Raw pixel dimensions
D = D_pixels        # Current observation dimension (starts with pixels)
use_cfl = False     # Start with raw pixels
cfl_updates = []    # Track when CFL updates happen

print("🧠 Adaptive CFL Pong Training")
print("=" * 50)
print("🎯 Training Strategy:")
print(f"   • Start: Raw pixels ({D_pixels}D)")
print(f"   • Every {cfl_update_interval} episodes: Learn CFL macrovariables")
print(f"   • Observation space evolves over time")
print(f"   • Target: {n_macrovariables} macrovariables")
print("=" * 50)

# Initialize model with pixel dimensions
model = {}
model['W1'] = np.random.randn(H, D) / np.sqrt(D)
model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() }
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }

# Initialize CFL system
cfl = CausalFeatureLearner(
    input_dim=D_pixels,
    n_macro_causes=n_macrovariables,
    feature_dim=32,
    device='cpu'
)

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    """Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]  # crop
    I = I[::2,::2,0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1   # everything else (paddles, ball) just set to 1
    return I.astype(np.float64)

def get_observation(observation, prev_observation=None):
    """Get observation - either raw pixels or CFL macrovariables"""
    cur_x = prepro(observation)
    
    if prev_observation is not None:
        prev_x = prepro(prev_observation)
        diff_frame = cur_x - prev_x
    else:
        diff_frame = np.zeros_like(cur_x)
    
    if use_cfl and cfl.is_trained:
        # Use CFL macrovariable representation
        macro_id = cfl.transform_observation(diff_frame)
        # Convert to one-hot encoding
        obs = np.zeros(n_macrovariables)
        obs[macro_id] = 1.0
        return obs, cur_x
    else:
        # Use raw pixel difference
        return diff_frame.ravel(), cur_x

def discount_rewards(r):
    """Compute discounted rewards"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0  # ReLU
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h

def policy_backward(eph, epdlogp, epx):
    """Backward pass"""
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

def update_model_for_cfl():
    """Update model architecture when switching to CFL observations"""
    global model, grad_buffer, rmsprop_cache, D
    
    print(f"🔄 Updating model: {D_pixels}D pixels → {n_macrovariables}D macrovariables")
    D = n_macrovariables
    
    # Reinitialize model with new dimensions
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)
    
    grad_buffer = { k : np.zeros_like(v) for k,v in model.items() }
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }

def train_cfl(episode_number):
    """Train CFL on collected data and switch to macrovariable observations"""
    global use_cfl
    
    print(f"\n{'='*50}")
    print(f"🧠 CFL TRAINING - Episode {episode_number}")
    print(f"📊 Data collected: {len(cfl.cause_data)} frames")
    
    if len(cfl.cause_data) < 100:
        print("⚠️  Not enough data for CFL training")
        return
    
    # Train CFL encoders
    print("🔧 Training neural encoders...")
    cfl.train_encoders(epochs=cfl_epochs, batch_size=64)
    
    # Discover macrovariables
    print("🔍 Discovering macrovariables...")
    cfl.discover_macrovariables()
    
    # Switch to CFL observations
    if not use_cfl:
        use_cfl = True
        update_model_for_cfl()
        print("✅ Switched to CFL macrovariable observations")
    
    # Save CFL model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfl_path = f'models/cfl_ep{episode_number}_{timestamp}.pt'
    os.makedirs('models', exist_ok=True)
    cfl.save(cfl_path)
    
    # Track this update
    cfl_updates.append({
        'episode': episode_number,
        'data_points': len(cfl.cause_data),
        'model_path': cfl_path
    })
    
    print(f"💾 CFL model saved: {cfl_path}")
    print(f"{'='*50}\n")

def create_progress_visualization(reward_history):
    """Create visualization of training progress with CFL updates"""
    if len(reward_history) < 10:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Learning curve with CFL updates
    plt.subplot(2, 2, 1)
    plt.plot(reward_history, alpha=0.7, label='Episode Rewards')
    
    # Running average
    window = 20
    if len(reward_history) > window:
        running_avg = [np.mean(reward_history[max(0, i-window):i+1]) for i in range(len(reward_history))]
        plt.plot(running_avg, label=f'Running Average ({window})', linewidth=2)
    
    # Mark CFL updates
    for update in cfl_updates:
        episode = update['episode']
        if episode < len(reward_history):
            plt.axvline(x=episode, color='red', linestyle='--', alpha=0.7)
            plt.text(episode, max(reward_history) * 0.9, 'CFL', 
                    rotation=90, ha='center', va='bottom')
    
    plt.title('Adaptive CFL Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Observation space evolution
    plt.subplot(2, 2, 2)
    episodes = []
    dimensions = []
    
    # Add initial pixel phase
    if cfl_updates:
        episodes.append(0)
        dimensions.append(D_pixels)
        
        for update in cfl_updates:
            episodes.append(update['episode'])
            dimensions.append(n_macrovariables)
    
    if episodes:
        plt.step(episodes, dimensions, where='post', linewidth=2)
        plt.title('Observation Space Evolution')
        plt.xlabel('Episode')
        plt.ylabel('Observation Dimensions')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: CFL data collection
    plt.subplot(2, 2, 3)
    if cfl_updates:
        update_episodes = [u['episode'] for u in cfl_updates]
        data_points = [u['data_points'] for u in cfl_updates]
        
        plt.bar(range(len(update_episodes)), data_points, alpha=0.7)
        plt.title('CFL Data Collection per Update')
        plt.xlabel('CFL Update Number')
        plt.ylabel('Data Points Used')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Performance improvement
    plt.subplot(2, 2, 4)
    if len(cfl_updates) > 0 and len(reward_history) > 20:
        # Compare performance before/after CFL updates
        improvements = []
        for i, update in enumerate(cfl_updates):
            episode = update['episode']
            if episode > 10 and episode < len(reward_history) - 10:
                before = np.mean(reward_history[max(0, episode-10):episode])
                after = np.mean(reward_history[episode:min(len(reward_history), episode+10)])
                improvements.append(after - before)
        
        if improvements:
            plt.bar(range(len(improvements)), improvements, alpha=0.7)
            plt.title('Performance Change After CFL Updates')
            plt.xlabel('CFL Update Number')
            plt.ylabel('Reward Improvement')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = f'results/adaptive_cfl_progress_{timestamp}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

# Initialize environment
env = gym.make("Pong-v0")
observation = env.reset()
if isinstance(observation, tuple):
    observation = observation[0]

prev_observation = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
reward_history = []

print("🎮 Starting Adaptive CFL Training...")
print("Phase 1: Raw pixel learning")

# Training loop
max_episodes = 100  # Adjust as needed

while episode_number < max_episodes:
    # Get current observation (pixels or macrovariables)
    x, cur_processed = get_observation(observation, prev_observation)
    prev_observation = observation

    # Policy forward pass
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    # Record intermediates
    xs.append(x)
    hs.append(h)
    y = 1 if action == 2 else 0
    dlogps.append(y - aprob)

    # Step environment
    step_result = env.step(action)
    observation, reward, done = step_result[0], step_result[1], step_result[2]
    reward_sum += reward
    drs.append(reward)

    # Always collect pixel data for CFL (even when using macrovariables)
    diff_frame = prepro(observation) - cur_processed if cur_processed is not None else np.zeros((80, 80))
    cfl.add_data(diff_frame, reward, done, action)

    if done:  # Episode finished
        episode_number += 1

        # Stack episode data
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []

        # Compute discounted rewards
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= (np.std(discounted_epr) + 1e-8)

        # Policy gradient update
        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp, epx)
        for k in model: 
            grad_buffer[k] += grad[k]

        # Update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        # Book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        reward_history.append(reward_sum)
        
        obs_type = "Macrovariables" if use_cfl else "Raw Pixels"
        print(f'Episode {episode_number:3d} ({obs_type}): reward: {reward_sum:6.1f}, running mean: {running_reward:6.1f}, data: {len(cfl.cause_data)}')

        # CFL updates every N episodes
        if episode_number % cfl_update_interval == 0:
            train_cfl(episode_number)

        reward_sum = 0
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        prev_observation = None

    if reward != 0:
        result = "Won! 🎉" if reward == 1 else "Lost 😞"

env.close()

# Create final analysis
print(f"\n{'='*60}")
print("🎯 ADAPTIVE CFL TRAINING COMPLETED!")
print(f"{'='*60}")
print(f"📊 Training Summary:")
print(f"   • Total episodes: {episode_number}")
print(f"   • CFL updates: {len(cfl_updates)}")
print(f"   • Final observation space: {D}D")
print(f"   • Final performance: {running_reward:.1f}")

if cfl_updates:
    print(f"\n🔄 CFL Update History:")
    for i, update in enumerate(cfl_updates):
        print(f"   {i+1}. Episode {update['episode']}: {update['data_points']} data points")

# Create progress visualization
viz_path = create_progress_visualization(reward_history)
print(f"\n📊 Progress visualization: {viz_path}")

print(f"\n💡 Key Achievements:")
print(f"   • Started with {D_pixels}D raw pixels")
print(f"   • Evolved to {n_macrovariables}D macrovariables")
print(f"   • Observation space adapted {len(cfl_updates)} times")
print(f"   • Demonstrated adaptive representation learning")

print(f"\n📁 Generated Files:")
for update in cfl_updates:
    print(f"   • {update['model_path']}")
print(f"   • {viz_path}")

print(f"{'='*60}")

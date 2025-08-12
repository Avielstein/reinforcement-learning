"""
CFL-Enhanced Pong from Pixels using Policy Gradients
Combines Karpathy's Policy Gradients with Causal Feature Learning for feature reduction
"""

import numpy as np
import pickle
import gym
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add CFL module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cfl.causal_feature_learner import CausalFeatureLearner

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# CFL hyperparameters
cfl_data_collection_episodes = 50  # Episodes to collect data for CFL training (reduced for demo)
n_macro_causes = 16  # Number of macrovariables to discover
cfl_feature_dim = 64  # Dimension of CFL features
use_cfl_features = False  # Start with raw features, switch after CFL training

# model initialization
D_raw = 80 * 80 # raw input dimensionality: 80x80 grid
D_cfl = cfl_feature_dim  # CFL feature dimensionality
D = D_raw  # Start with raw features

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

# Initialize CFL
cfl = CausalFeatureLearner(
    input_dim=D_raw,
    n_macro_causes=n_macro_causes,
    feature_dim=cfl_feature_dim,
    device='cpu'
)

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float64)

def get_features(observation, prev_observation=None):
    """Get features from observation - either raw or CFL-processed"""
    # Preprocess observation
    cur_x = prepro(observation)
    
    if prev_observation is not None:
        prev_x = prepro(prev_observation)
        diff_frame = cur_x - prev_x
    else:
        diff_frame = np.zeros_like(cur_x)
    
    if use_cfl_features and cfl.is_trained:
        # Use CFL macrovariable features
        macro_label = cfl.transform_observation(diff_frame)
        features = cfl.get_macrovariable_features(macro_label)
        return features, cur_x
    else:
        # Use raw difference frame
        return diff_frame.ravel(), cur_x

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp, epx):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backprop prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

def reinitialize_model_for_cfl():
    """Reinitialize model weights for CFL features"""
    global model, grad_buffer, rmsprop_cache, D
    
    print("Reinitializing model for CFL features...")
    D = D_cfl
    
    # Reinitialize with new dimensions
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)
    
    # Reset buffers
    grad_buffer = { k : np.zeros_like(v) for k,v in model.items() }
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }
    
    print(f"Model reinitialized: Input dim {D_raw} -> {D}")

def train_cfl_model():
    """Train the CFL model on collected data"""
    print(f"\nTraining CFL model on {len(cfl.cause_data)} data points...")
    
    # Train encoders
    cfl.train_encoders(epochs=100, batch_size=64)
    
    # Discover macrovariables
    cfl.discover_macrovariables()
    
    # Save CFL model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfl_path = f'models/cfl_model_{timestamp}.pt'
    cfl.save(cfl_path)
    
    # Visualize results
    viz_path = f'results/cfl_visualization_{timestamp}.png'
    cfl.visualize_macrovariables(save_path=viz_path)
    
    print(f"CFL training completed! Model saved to {cfl_path}")
    return cfl_path

def save_model(episode, reward_sum, suffix=""):
    """Save model and training statistics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = f'models/pong_cfl_{suffix}_{timestamp}_ep{episode}.p'
    os.makedirs('models', exist_ok=True)
    pickle.dump(model, open(model_path, 'wb'))
    
    # Save training stats
    stats = {
        'episode': episode,
        'reward_sum': reward_sum,
        'running_reward': running_reward,
        'reward_history': reward_history,
        'use_cfl_features': use_cfl_features,
        'cfl_trained': cfl.is_trained
    }
    stats_path = f'models/stats_cfl_{suffix}_{timestamp}_ep{episode}.p'
    pickle.dump(stats, open(stats_path, 'wb'))
    
    print(f'Model saved: {model_path}')
    return model_path

def plot_training_progress():
    """Plot and save training progress"""
    if len(reward_history) < 2:
        return
        
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(reward_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Mark CFL transition point
    if use_cfl_features and len(reward_history) > cfl_data_collection_episodes:
        plt.axvline(x=cfl_data_collection_episodes, color='red', linestyle='--', 
                   label='CFL Transition')
        plt.legend()
    
    plt.subplot(1, 3, 2)
    # Running average
    window = min(100, len(reward_history))
    running_avg = [np.mean(reward_history[max(0, i-window):i+1]) for i in range(len(reward_history))]
    plt.plot(running_avg)
    plt.title(f'Running Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    if use_cfl_features and len(reward_history) > cfl_data_collection_episodes:
        plt.axvline(x=cfl_data_collection_episodes, color='red', linestyle='--', 
                   label='CFL Transition')
        plt.legend()
    
    plt.subplot(1, 3, 3)
    # Compare pre/post CFL performance
    if use_cfl_features and len(reward_history) > cfl_data_collection_episodes + 50:
        pre_cfl = reward_history[:cfl_data_collection_episodes]
        post_cfl = reward_history[cfl_data_collection_episodes:]
        
        pre_avg = np.mean(pre_cfl[-50:]) if len(pre_cfl) >= 50 else np.mean(pre_cfl)
        post_avg = np.mean(post_cfl[:50]) if len(post_cfl) >= 50 else np.mean(post_cfl)
        
        plt.bar(['Pre-CFL', 'Post-CFL'], [pre_avg, post_avg])
        plt.title('Performance Comparison')
        plt.ylabel('Average Reward')
        plt.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/training_progress_cfl_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Initialize environment and training variables
env = gym.make("Pong-v0")
observation = env.reset()
# Handle new gym API that returns (obs, info) tuple
if isinstance(observation, tuple):
    observation = observation[0]
prev_observation = None
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
reward_history = []

print("Starting CFL-Enhanced Pong training with Policy Gradients...")
print(f"Phase 1: Collecting data for CFL training ({cfl_data_collection_episodes} episodes)")
print(f"Phase 2: Training CFL model")
print(f"Phase 3: Training with CFL features")
print(f"Hidden units: {H}, Batch size: {batch_size}, Learning rate: {learning_rate}")
print(f"CFL macrovariables: {n_macro_causes}, Feature dim: {cfl_feature_dim}")
print("=" * 80)

while True:
    if render: env.render()

    # Get features (raw or CFL-processed)
    x, cur_processed = get_features(observation, prev_observation)
    prev_observation = observation

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation features
    hs.append(h) # hidden state
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken

    # step the environment and get new measurements
    step_result = env.step(action)
    observation, reward, done = step_result[0], step_result[1], step_result[2]
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    # Collect data for CFL training (only during data collection phase)
    if not use_cfl_features and episode_number < cfl_data_collection_episodes:
        # Add data point for CFL training
        diff_frame = prepro(observation) - cur_processed if cur_processed is not None else np.zeros((80, 80))
        cfl.add_data(diff_frame, reward, done, action)

    if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp, epx)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        reward_history.append(reward_sum)
        
        phase_indicator = "CFL-Data" if not use_cfl_features else "CFL-Train"
        print(f'Episode {episode_number:4d} ({phase_indicator}): reward: {reward_sum:6.1f}, running mean: {running_reward:6.1f}')
        
        # Check if we should train CFL and switch to CFL features
        if episode_number == cfl_data_collection_episodes and not use_cfl_features:
            print(f"\n{'='*80}")
            print("DATA COLLECTION PHASE COMPLETED")
            print(f"Collected {len(cfl.cause_data)} data points for CFL training")
            print("Starting CFL training...")
            print(f"{'='*80}")
            
            # Train CFL model
            cfl_model_path = train_cfl_model()
            
            # Switch to CFL features
            use_cfl_features = True
            reinitialize_model_for_cfl()
            
            print(f"\n{'='*80}")
            print("SWITCHING TO CFL FEATURES")
            print(f"Feature dimensionality reduced: {D_raw} -> {D_cfl}")
            print(f"Compression ratio: {D_raw/D_cfl:.1f}x")
            print("Continuing training with CFL macrovariables...")
            print(f"{'='*80}\n")
        
        # Save model periodically
        if episode_number % 100 == 0:
            suffix = "data_collection" if not use_cfl_features else "cfl_enhanced"
            save_model(episode_number, reward_sum, suffix)
            plot_training_progress()
        
        # Save checkpoint
        if episode_number % 1000 == 0:
            suffix = "data_collection" if not use_cfl_features else "cfl_enhanced"
            final_path = save_model(episode_number, reward_sum, suffix)
            print(f'Checkpoint saved at episode {episode_number}')
        
        reward_sum = 0
        observation = env.reset() # reset env
        # Handle new gym API that returns (obs, info) tuple
        if isinstance(observation, tuple):
            observation = observation[0]
        prev_observation = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        result = "Won!" if reward == 1 else "Lost..."
        phase = "Data Collection" if not use_cfl_features else "CFL Training"
        print(f'Episode {episode_number} ({phase}): {result} (reward: {reward})')

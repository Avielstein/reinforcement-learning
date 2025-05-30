## 6. Comprehensive Long Training Session

This section provides a thorough training session with detailed progress tracking, automatic model saving, and comprehensive visualization.

```python
# Long training session with comprehensive tracking and model saving

import time
from collections import deque
import os

# Training configuration
TRAINING_EPISODES = 2000
EVAL_INTERVAL = 50
SAVE_INTERVAL = 200
PATIENCE = 300  # Early stopping patience
MOVING_AVERAGE_WINDOW = 100

# Create directories for saving models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('training_plots', exist_ok=True)

print(f"Starting comprehensive training for {TRAINING_EPISODES} episodes...")
print(f"Model checkpoints will be saved every {SAVE_INTERVAL} episodes")
print(f"Evaluation every {EVAL_INTERVAL} episodes")
print("="*60)

# Initialize learner
long_learner = DotFollowLearner('circular')

# Training metrics tracking
training_metrics = {
    'episodes': [],
    'rewards': [],
    'distances': [],
    'episode_lengths': [],
    'eval_rewards': [],
    'eval_episodes': [],
    'best_reward': -np.inf,
    'best_episode': 0,
    'training_time': []
}

# Moving averages for smoothing
reward_ma = deque(maxlen=MOVING_AVERAGE_WINDOW)
distance_ma = deque(maxlen=MOVING_AVERAGE_WINDOW)

# Training loop
start_time = time.time()
best_eval_reward = -np.inf
episodes_without_improvement = 0

for episode in range(TRAINING_EPISODES):
    episode_start = time.time()
    
    # Training step
    long_learner.train_step()
    
    episode_time = time.time() - episode_start
    training_metrics['training_time'].append(episode_time)
    
    # Get current episode metrics
    if long_learner.ep_returns:
        current_reward = list(long_learner.ep_returns)[-1]
        reward_ma.append(current_reward)
        
        # Get performance metrics
        metrics = long_learner.get_performance_metrics()
        if metrics:
            distance_ma.append(metrics['mean_target_distance'])
            
            # Store metrics
            training_metrics['episodes'].append(episode)
            training_metrics['rewards'].append(np.mean(reward_ma))
            training_metrics['distances'].append(np.mean(distance_ma))
            training_metrics['episode_lengths'].append(metrics.get('mean_episode_length', 0))
    
    # Periodic evaluation and logging
    if episode % EVAL_INTERVAL == 0 and episode > 0:
        # Evaluation on multiple patterns
        eval_rewards = []
        eval_patterns = ['circular', 'figure8', 'random_walk']
        
        for pattern in eval_patterns:
            env = DotFollowEnv(pattern)
            obs = env.reset()
            total_reward = 0
            
            for _ in range(200):  # Longer evaluation episodes
                action = long_learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            
            eval_rewards.append(total_reward)
        
        avg_eval_reward = np.mean(eval_rewards)
        training_metrics['eval_rewards'].append(avg_eval_reward)
        training_metrics['eval_episodes'].append(episode)
        
        # Check for improvement
        if avg_eval_reward > best_eval_reward:
            best_eval_reward = avg_eval_reward
            training_metrics['best_reward'] = avg_eval_reward
            training_metrics['best_episode'] = episode
            episodes_without_improvement = 0
            
            # Save best model
            long_learner.save_model('models/best_dot_follow_model.pt')
            print(f"🏆 New best model saved! Eval reward: {avg_eval_reward:.2f}")
        else:
            episodes_without_improvement += EVAL_INTERVAL
        
        # Progress logging
        elapsed_time = time.time() - start_time
        avg_reward = np.mean(reward_ma) if reward_ma else 0
        avg_distance = np.mean(distance_ma) if distance_ma else 0
        
        print(f"Episode {episode:4d} | "
              f"Reward: {avg_reward:6.2f} | "
              f"Distance: {avg_distance:5.2f} | "
              f"Eval: {avg_eval_reward:6.2f} | "
              f"Time: {elapsed_time/60:4.1f}m | "
              f"Best: {best_eval_reward:6.2f}@{training_metrics['best_episode']}")
    
    # Save checkpoint
    if episode % SAVE_INTERVAL == 0 and episode > 0:
        checkpoint_path = f'models/checkpoint_episode_{episode}.pt'
        long_learner.save_model(checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Early stopping check
    if episodes_without_improvement >= PATIENCE:
        print(f"\n🛑 Early stopping triggered after {episodes_without_improvement} episodes without improvement")
        break

total_training_time = time.time() - start_time
print(f"\n🎯 Training completed!")
print(f"Total time: {total_training_time/60:.1f} minutes")
print(f"Episodes trained: {episode + 1}")
print(f"Best evaluation reward: {best_eval_reward:.2f} at episode {training_metrics['best_episode']}")
print(f"Average time per episode: {np.mean(training_metrics['training_time']):.3f}s")

# Load the best model for final evaluation
long_learner.load_model('models/best_dot_follow_model.pt')
print("✅ Best model loaded for comprehensive evaluation")
```

### Comprehensive Training Visualization

Now let's create detailed visualizations of the training progress.

```python
# Create comprehensive training visualizations

fig = plt.figure(figsize=(20, 12))

# 1. Training Rewards Over Time
ax1 = plt.subplot(2, 3, 1)
if training_metrics['rewards']:
    plt.plot(training_metrics['episodes'], training_metrics['rewards'], 'b-', linewidth=2, alpha=0.7, label='Moving Average')
    plt.scatter(training_metrics['eval_episodes'], training_metrics['eval_rewards'], 
                c='red', s=50, alpha=0.8, label='Evaluation', zorder=5)
    plt.axhline(y=training_metrics['best_reward'], color='green', linestyle='--', 
                alpha=0.7, label=f'Best: {training_metrics["best_reward"]:.2f}')
    plt.title('Training Progress: Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 2. Target Distance Over Time
ax2 = plt.subplot(2, 3, 2)
if training_metrics['distances']:
    plt.plot(training_metrics['episodes'], training_metrics['distances'], 'r-', linewidth=2, alpha=0.7)
    plt.title('Average Distance to Target')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)

# 3. Episode Lengths
ax3 = plt.subplot(2, 3, 3)
if training_metrics['episode_lengths']:
    plt.plot(training_metrics['episodes'], training_metrics['episode_lengths'], 'g-', linewidth=2, alpha=0.7)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps per Episode')
    plt.grid(True, alpha=0.3)

# 4. Training Time per Episode
ax4 = plt.subplot(2, 3, 4)
if training_metrics['training_time']:
    # Smooth the training time data
    smoothed_times = np.convolve(training_metrics['training_time'], 
                                np.ones(min(50, len(training_metrics['training_time'])))/min(50, len(training_metrics['training_time'])), 
                                mode='valid')
    plt.plot(smoothed_times, 'm-', linewidth=2, alpha=0.7)
    plt.title('Training Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)

# 5. Evaluation Performance Comparison
ax5 = plt.subplot(2, 3, 5)
if training_metrics['eval_rewards']:
    plt.plot(training_metrics['eval_episodes'], training_metrics['eval_rewards'], 'ro-', 
             linewidth=2, markersize=6, alpha=0.8)
    plt.title('Evaluation Performance')
    plt.xlabel('Episode')
    plt.ylabel('Average Evaluation Reward')
    plt.grid(True, alpha=0.3)

# 6. Training Summary Statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
TRAINING SUMMARY

Total Episodes: {len(training_metrics['training_time'])}
Total Time: {total_training_time/60:.1f} minutes
Avg Time/Episode: {np.mean(training_metrics['training_time']):.3f}s

Best Evaluation Reward: {training_metrics['best_reward']:.2f}
Best Episode: {training_metrics['best_episode']}

Final Avg Reward: {training_metrics['rewards'][-1] if training_metrics['rewards'] else 'N/A':.2f}
Final Avg Distance: {training_metrics['distances'][-1] if training_metrics['distances'] else 'N/A':.2f}

Model Saved: models/best_dot_follow_model.pt
"""
ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('training_plots/comprehensive_training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Comprehensive training analysis saved to 'training_plots/comprehensive_training_analysis.png'")
```

### Final Model Performance Evaluation

Let's thoroughly test the trained model across all movement patterns.

```python
# Comprehensive evaluation of the trained model

eval_patterns = ['circular', 'figure8', 'random_walk', 'zigzag', 'spiral']
eval_results = {}

print("🧪 Comprehensive Model Evaluation")
print("="*50)

for pattern in eval_patterns:
    print(f"\nTesting on {pattern} pattern...")
    
    pattern_rewards = []
    pattern_distances = []
    pattern_episode_lengths = []
    
    for trial in range(10):  # 10 trials per pattern
        env = DotFollowEnv(pattern)
        obs = env.reset()
        
        total_reward = 0
        distances = []
        steps = 0
        
        for step in range(500):  # Max 500 steps per episode
            action = long_learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = env.step(action)
            
            total_reward += reward
            distance = np.linalg.norm(env.position - env.target.position)
            distances.append(distance)
            steps += 1
            
            if done:
                break
        
        pattern_rewards.append(total_reward)
        pattern_distances.append(np.mean(distances))
        pattern_episode_lengths.append(steps)
    
    # Store results
    eval_results[pattern] = {
        'rewards': pattern_rewards,
        'distances': pattern_distances,
        'episode_lengths': pattern_episode_lengths,
        'mean_reward': np.mean(pattern_rewards),
        'std_reward': np.std(pattern_rewards),
        'mean_distance': np.mean(pattern_distances),
        'std_distance': np.std(pattern_distances),
        'mean_length': np.mean(pattern_episode_lengths),
        'std_length': np.std(pattern_episode_lengths)
    }
    
    print(f"  Reward: {eval_results[pattern]['mean_reward']:.2f} ± {eval_results[pattern]['std_reward']:.2f}")
    print(f"  Distance: {eval_results[pattern]['mean_distance']:.2f} ± {eval_results[pattern]['std_distance']:.2f}")
    print(f"  Episode Length: {eval_results[pattern]['mean_length']:.1f} ± {eval_results[pattern]['std_length']:.1f}")

# Create evaluation comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Reward comparison
ax1 = axes[0, 0]
patterns = list(eval_results.keys())
mean_rewards = [eval_results[p]['mean_reward'] for p in patterns]
std_rewards = [eval_results[p]['std_reward'] for p in patterns]

bars1 = ax1.bar(patterns, mean_rewards, yerr=std_rewards, capsize=5, 
                color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
ax1.set_title('Average Reward by Movement Pattern')
ax1.set_ylabel('Reward')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Distance comparison
ax2 = axes[0, 1]
mean_distances = [eval_results[p]['mean_distance'] for p in patterns]
std_distances = [eval_results[p]['std_distance'] for p in patterns]

bars2 = ax2.bar(patterns, mean_distances, yerr=std_distances, capsize=5,
                color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
ax2.set_title('Average Distance to Target')
ax2.set_ylabel('Distance')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# Episode length comparison
ax3 = axes[1, 0]
mean_lengths = [eval_results[p]['mean_length'] for p in patterns]
std_lengths = [eval_results[p]['std_length'] for p in patterns]

bars3 = ax3.bar(patterns, mean_lengths, yerr=std_lengths, capsize=5,
                color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
ax3.set_title('Average Episode Length')
ax3.set_ylabel('Steps')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Performance ranking
ax4 = axes[1, 1]
# Rank by reward (higher is better)
ranked_patterns = sorted(patterns, key=lambda p: eval_results[p]['mean_reward'], reverse=True)
ranked_rewards = [eval_results[p]['mean_reward'] for p in ranked_patterns]

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(ranked_patterns)))
bars4 = ax4.barh(range(len(ranked_patterns)), ranked_rewards, color=colors)
ax4.set_yticks(range(len(ranked_patterns)))
ax4.set_yticklabels(ranked_patterns)
ax4.set_title('Performance Ranking (by Reward)')
ax4.set_xlabel('Average Reward')
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for i, (pattern, reward) in enumerate(zip(ranked_patterns, ranked_rewards)):
    ax4.text(reward + 0.5, i, f'{reward:.1f}', va='center')

plt.tight_layout()
plt.savefig('training_plots/final_evaluation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📈 Final evaluation results saved to 'training_plots/final_evaluation_results.png'")
print("\n🎉 Comprehensive training and evaluation completed!")
print(f"Best model saved at: models/best_dot_follow_model.pt")
print(f"Training plots saved in: training_plots/")

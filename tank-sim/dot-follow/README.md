# Dot Follow Reinforcement Learning

Train a fish agent to follow a moving target around the tank. This project demonstrates dynamic target tracking with multiple movement patterns, providing a more challenging learning scenario than static objectives.

## Overview

The dot-follow environment extends the basic fish tank simulation by introducing a **moving target** that the fish must learn to track. Unlike the fish-to-center project where the goal is static, here the fish must continuously adapt its behavior to follow a target that moves in various patterns.

## Key Features

### Dynamic Target Tracking
- **Moving Target**: Red dot that moves in predefined patterns around the tank
- **Real-time Adaptation**: Fish must continuously adjust its strategy as the target moves
- **Multiple Movement Patterns**: 5 different target movement behaviors to master

### Enhanced Learning Environment
- **6D Observation Space**: Fish position, velocity, and target position (all relative to center)
- **Reward Shaping**: Proximity-based rewards with bonuses for close following
- **Reduced Distractions**: Fewer and weaker water currents to focus on tracking behavior
- **Interactive Training**: Change movement patterns during training with button controls

### Movement Patterns

1. **Circular**: Target moves in a circle around the tank center
2. **Figure-8**: Target follows a figure-8 pattern
3. **Random Walk**: Target moves with smooth random direction changes
4. **Zigzag**: Target moves in a zigzag pattern across the tank
5. **Spiral**: Target moves in an expanding spiral (resets when reaching edges)

## Project Structure

```
dot-follow/
├── README.md                           # This file
├── train_dot_follow.ipynb             # Main training notebook
├── dot_follow_environment.py          # Environment and target movement logic
├── dot_follow_trainer.py              # A2C trainer specialized for dot following
└── dot_follow_visualization.py        # Interactive training visualization
```

## Quick Start

### Prerequisites
```bash
pip install torch gymnasium matplotlib numpy jupyter
```

### Basic Training
```bash
cd tank-sim/dot-follow
jupyter notebook train_dot_follow.ipynb
```

### Interactive Training with Visualization
```python
from dot_follow_visualization import run_dot_follow_training

# Start training with real-time visualization
learner = run_dot_follow_training('circular')
```

## Usage Examples

### Basic Environment Usage
```python
from dot_follow_environment import DotFollowEnv

# Create environment with circular target movement
env = DotFollowEnv('circular')
obs = env.reset()

# Take a step
action = [0.5, -0.3]  # Thrust right and down
next_obs, reward, done, info = env.step(action)

# Get current target position
target_pos = env.get_target_position()
```

### Training a Model
```python
from dot_follow_trainer import DotFollowLearner

# Create learner
learner = DotFollowLearner('figure8')

# Train for 100 episodes
for _ in range(100):
    learner.train_step()

# Save best model
learner.save_model('my_dot_follow_model.pt')
```

### Testing Different Patterns
```python
from dot_follow_visualization import test_dot_follow_model

# Test trained model on multiple patterns
test_dot_follow_model(learner, 
                     num_episodes=3, 
                     movement_patterns=['circular', 'figure8', 'random_walk'])
```

## Environment Details

### Observation Space (6 dimensions)
- **Fish Position**: (x, y) relative to tank center, normalized to [-1, 1]
- **Fish Velocity**: (vx, vy) normalized by maximum velocity
- **Target Position**: (x, y) relative to tank center, normalized to [-1, 1]

### Action Space (2 dimensions)
- **Thrust Forces**: Continuous values in x and y directions, range [-1, 1]

### Reward Function
The reward encourages the fish to stay close to the moving target:

```python
# Primary reward: inverse distance to target
proximity_reward = 2.0 * (1.0 - target_distance / max_possible_distance)

# Bonuses for being very close
if target_distance < 5.0:
    proximity_reward += 3.0
elif target_distance < 10.0:
    proximity_reward += 1.5
elif target_distance < 15.0:
    proximity_reward += 0.5

# Small penalties for high velocity and wall proximity
velocity_penalty = -0.005 * velocity_magnitude
wall_penalty = -0.3 if near_walls else 0.0

# Bonus for getting closer to target
if getting_closer:
    proximity_reward += 0.2

total_reward = proximity_reward + velocity_penalty + wall_penalty
```

## Training Features

### Interactive Visualization
- **Real-time Training**: Watch the fish learn to follow the target
- **Pattern Switching**: Change target movement patterns during training using buttons
- **Performance Metrics**: Live plots of reward and distance to target
- **Trail Visualization**: See both fish and target movement trails

### Curriculum Learning
Train on progressively more difficult patterns:
```python
curriculum = ['circular', 'figure8', 'zigzag', 'random_walk']
```

### Model Persistence
- Automatic saving of best-performing models
- Easy loading and testing of trained models
- Performance comparison across different patterns

## Advanced Usage

### Custom Movement Patterns
Extend the `MovingTarget` class to create custom movement patterns:

```python
class MovingTarget:
    def step(self):
        if self.movement_pattern == 'my_custom_pattern':
            # Implement your custom movement logic here
            self.position[0] = custom_x_calculation()
            self.position[1] = custom_y_calculation()
```

### Hyperparameter Tuning
Key parameters to experiment with:

```python
# In dot_follow_environment.py
target.speed = 15.0          # Target movement speed
target.radius = 25.0         # Pattern size

# In utils/constants.py
POLICY_LR = 1e-3            # Learning rate
EPISODE_LEN = 200           # Episode length
BATCH_SIZE = 512            # Training batch size
```

### Multi-Agent Extensions
The environment can be extended to support multiple fish following the same target:

```python
# Future enhancement idea
class MultiAgentDotFollowEnv:
    def __init__(self, num_fish=3, movement_pattern='circular'):
        self.fish_agents = [FishAgent() for _ in range(num_fish)]
        self.target = MovingTarget(movement_pattern)
```

## Performance Tips

1. **Start Simple**: Begin with circular patterns before moving to random walk
2. **Monitor Distance**: Watch the average distance to target metric
3. **Adjust Learning Rate**: Lower rates for more stable learning
4. **Pattern Progression**: Use curriculum learning for better generalization
5. **Visualization**: Use interactive training to understand behavior

## Comparison with Fish-to-Center

| Aspect | Fish-to-Center | Dot-Follow |
|--------|----------------|------------|
| **Target** | Static center point | Moving target |
| **Difficulty** | Moderate | Higher |
| **Observation Space** | 4D (position + velocity) | 6D (+ target position) |
| **Learning Challenge** | Reach and stay | Track and follow |
| **Reward Structure** | Distance to center | Distance to moving target |
| **Generalization** | Single objective | Multiple movement patterns |

## Research Applications

This environment is suitable for studying:
- **Dynamic Target Tracking**: How agents adapt to moving objectives
- **Curriculum Learning**: Progressive difficulty in movement patterns
- **Transfer Learning**: Knowledge transfer between different patterns
- **Multi-Modal Learning**: Handling different types of target behavior
- **Continuous Control**: Smooth following behavior in continuous action spaces

## Troubleshooting

### Common Issues

**Fish not following target:**
- Check if target movement speed is too fast
- Increase proximity rewards
- Reduce water current strength

**Training not converging:**
- Start with simpler patterns (circular)
- Reduce learning rate
- Increase episode length

**Visualization not working:**
- Ensure matplotlib backend supports animation
- Try different Python environments
- Check for GUI display issues

## Future Enhancements

- **Predictive Following**: Fish learns to anticipate target movement
- **Multi-Target Scenarios**: Multiple targets with different priorities
- **Obstacle Avoidance**: Add obstacles between fish and target
- **Communication**: Multiple fish coordinating to track targets
- **Adaptive Patterns**: Target movement that responds to fish behavior

## Contributing

To extend this project:
1. Add new movement patterns in `MovingTarget.step()`
2. Implement new reward functions in `DotFollowEnv._reward()`
3. Create new visualization modes in `dot_follow_visualization.py`
4. Add experiment notebooks for specific research questions

## License

MIT License - feel free to use for research and educational purposes.

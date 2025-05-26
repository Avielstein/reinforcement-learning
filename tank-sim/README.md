# Fish Tank Reinforcement Learning

A comprehensive reinforcement learning environment where an AI agent learns to navigate a fish to the center of a tank while dealing with dynamic water currents. This project demonstrates modern RL techniques including Actor-Critic (A2C) algorithms with both Python/PyTorch implementation and interactive web visualization.

## Features

### Core RL Environment
- **Physics-based simulation** with realistic fish movement and water dynamics
- **Dynamic water currents** that change position and direction over time
- **Reward system** encouraging fish to stay near the center while maintaining stability
- **Configurable environment parameters** (tank size, current strength, episode length)

### Training Implementation
- **A2C (Advantage Actor-Critic)** algorithm with GAE (Generalized Advantage Estimation)
- **Actor-Critic neural networks** with continuous action spaces
- **Real-time training visualization** showing fish movement and learning progress
- **Model persistence** with automatic saving of best-performing policies

### Interactive Visualization
- **Live training monitoring** with matplotlib animations
- **Web-based simulator** with interactive controls and real-time charts
- **Performance metrics** tracking reward, distance to center, and learning progress
- **Multiple episode trajectory visualization** for trained agents

## Project Structure

```
tank-sim/
├── train-RL-fish.ipynb          # Main training notebook with examples
├── fish-tank-rl.html            # Interactive web-based simulator
├── utils/                       # Core implementation modules
│   ├── __init__.py              # Package initialization
│   ├── constants.py             # Environment and training parameters
│   ├── environment.py           # Fish tank environment and water currents
│   ├── models.py                # Neural network architectures (Actor-Critic)
│   ├── trainer.py               # A2C training algorithm implementation
│   └── visualization.py         # Training visualization and testing tools
├── *.pt                         # Saved model files (generated during training)
└── *.ipynb                      # Additional experiment notebooks
```

## Quick Start

### Prerequisites

```bash
pip install torch gymnasium matplotlib numpy jupyter
```

### Training a Fish Agent

1. **Open the main notebook:**
   ```bash
   jupyter notebook train-RL-fish.ipynb
   ```

2. **Run the training with visualization:**
   ```python
   # This will open a live visualization window
   learner = run_training_visualization()
   ```

3. **Test the trained model:**
   ```python
   test_trained_model(learner, num_episodes=5)
   ```

### Web-based Interactive Simulator

Open `fish-tank-rl.html` in your browser for an interactive experience with:
- Real-time learning visualization
- Adjustable environment parameters
- Speed controls and training metrics
- Q-learning implementation (simplified for web)

## Environment Details

### State Space (4 dimensions)
- **Position**: Fish position relative to tank center (normalized)
- **Velocity**: Fish velocity components (normalized)

### Action Space (2 dimensions)
- **Thrust forces**: Continuous values in x and y directions ([-1, 1])

### Reward Function
- **Center proximity**: Exponential reward for being close to center
- **Stability bonus**: Additional reward for low velocity (encourages hovering)
- **Wall penalty**: Negative reward for approaching tank boundaries
- **Special bonuses**: Extra rewards for being very close to center

### Water Currents
- **Dynamic positioning**: Currents slowly drift around the tank
- **Variable strength**: Each current has different force magnitudes
- **Directional changes**: Current directions evolve over time
- **Influence radius**: Force decreases with distance from current center

## Training Algorithm

The implementation uses **A2C (Advantage Actor-Critic)** with:

- **Policy Network**: Outputs continuous actions (fish thrust)
- **Value Network**: Estimates state values for advantage calculation
- **GAE**: Generalized Advantage Estimation for variance reduction
- **Gradient clipping**: Prevents training instability
- **Experience replay**: Trains on complete episodes

### Key Hyperparameters

```python
GAMMA = 0.99          # Discount factor
LAMBDA = 0.95         # GAE lambda
POLICY_LR = 1e-3      # Policy learning rate
VALUE_LR = 1e-3       # Value function learning rate
BATCH_SIZE = 512      # Training batch size
EPISODE_LEN = 200     # Maximum episode length
```

## Usage Examples

### Basic Training
```python
from utils import A2CLearner

# Create and train agent
learner = A2CLearner()
for _ in range(100):
    learner.train_step()

# Save trained model
learner.save_model('my_fish_policy.pt')
```

### Custom Environment
```python
from utils import FishTankEnv, WaterCurrent

# Create environment with custom settings
env = FishTankEnv()
env.currents = [WaterCurrent() for _ in range(3)]  # 3 currents

# Test environment
obs = env.reset()
action = [0.5, -0.3]  # Thrust right and down
next_obs, reward, done, info = env.step(action)
```

### Loading and Testing Models
```python
# Load a saved model
learner = A2CLearner()
learner.load_model('best_fish_policy.pt')

# Test performance
test_trained_model(learner, num_episodes=10)
```

## Customization

### Environment Parameters
Edit `utils/constants.py` to modify:
- Tank size and physics parameters
- Current count and strength
- Episode length and reward scaling
- Training hyperparameters

### Network Architecture
Modify `utils/models.py` to experiment with:
- Hidden layer sizes
- Activation functions
- Network depth
- Output distributions

### Training Algorithm
Adjust `utils/trainer.py` for:
- Different RL algorithms
- Custom advantage estimation
- Alternative update rules
- Exploration strategies

## Performance Tips

1. **Start simple**: Begin with fewer currents and shorter episodes
2. **Monitor training**: Use the visualization to watch learning progress
3. **Adjust learning rates**: Lower rates for more stable training
4. **Experiment with rewards**: Modify reward function for different behaviors
5. **Save frequently**: Best models are automatically saved during training

## Troubleshooting

### Common Issues

**Training not converging:**
- Reduce learning rates
- Increase episode length
- Simplify environment (fewer currents)

**Fish not reaching center:**
- Increase center proximity rewards
- Reduce current strength
- Check wall penalty settings

**Visualization not working:**
- Ensure matplotlib backend supports animation
- Try running in different environments (Jupyter vs terminal)

## Research Applications

This environment is suitable for:
- **RL algorithm comparison** (PPO, SAC, TD3, etc.)
- **Curriculum learning** experiments
- **Multi-agent scenarios** (multiple fish)
- **Transfer learning** studies
- **Continuous control** research

## Contributing

To extend this project:
1. Add new environment variants in `environment.py`
2. Implement additional RL algorithms in `trainer.py`
3. Create new visualization modes in `visualization.py`
4. Add experiment notebooks for specific research questions

## License

MIT License - feel free to use for research and educational purposes.

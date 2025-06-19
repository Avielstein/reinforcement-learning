# WaterWorld RAINBOW DQN

This project implements the RAINBOW DQN algorithm in a custom WaterWorld environment. RAINBOW combines several key improvements to DQN:

1. **Double DQN** - Reduces overestimation bias
2. **Dueling Networks** - Separates value and advantage estimation
3. **Prioritized Experience Replay** - Samples important experiences more frequently
4. **Multi-step Learning** - Uses n-step returns for better credit assignment
5. **Distributional RL** - Models the full return distribution instead of just the mean
6. **Noisy Networks** - Replaces epsilon-greedy exploration with learnable noise

## Environment

The WaterWorld environment features:
- An agent that moves in a 2D continuous space
- Good items (green) that provide positive rewards (+1)
- Bad items (red) that provide negative rewards (-1)
- 30 raycast sensors for perception
- Continuous movement with discrete action space

## Key Features

### RAINBOW Algorithm Components

- **Distributional RL**: Uses 51 atoms to model the value distribution
- **Noisy Networks**: Factorized Gaussian noise for exploration
- **Multi-step Learning**: 3-step returns for better temporal credit assignment
- **Prioritized Replay**: Importance sampling with priority updates
- **Dueling Architecture**: Separate value and advantage streams
- **Double DQN**: Action selection with online network, evaluation with target network

### Network Architecture

- **Input**: Sensor readings (120 dimensions: 30 sensors × 4 features each)
- **Hidden Layers**: [512, 512] with ReLU activation
- **Output**: Probability distributions over 51 atoms for each action
- **Noisy Layers**: Applied to final layers for exploration

### Training Configuration

- **Learning Rate**: 0.0005
- **Batch Size**: 32
- **Replay Buffer**: 50,000 experiences
- **Target Update**: Every 500 steps
- **Multi-step**: 3-step returns
- **Value Range**: [-10, 10] with 51 atoms

## Usage

### Training

```bash
# Basic training
python train_rainbow.py

# Custom training parameters
python train_rainbow.py --episodes 2000 --save-freq 200 --eval-freq 100 --verbose

# Training with specific save directory
python train_rainbow.py --save-dir my_models --episodes 1500
```

### Web Interface

```bash
# Start the web interface (if available)
python main.py
```

The web interface provides:
- Real-time visualization of training
- Interactive parameter adjustment
- Model loading and saving
- Performance metrics and charts

## File Structure

```
rainbow-experiments/waterworld-rainbow/
├── agent/
│   ├── rainbow.py          # RAINBOW DQN implementation
│   ├── networks.py         # Neural network architectures
│   ├── trainer.py          # Training manager
│   └── replay_buffer.py    # Prioritized experience replay
├── config/
│   ├── agent_config.py     # RAINBOW agent configuration
│   ├── environment_config.py # Environment settings
│   └── ui_config.py        # UI configuration
├── environment/
│   ├── waterworld.py       # Main environment
│   ├── entities.py         # Agent and items
│   ├── sensors.py          # Sensor system
│   └── physics.py          # Physics engine
├── server/                 # Web server components
├── static/
│   └── index.html          # Web interface
├── models/                 # Saved models directory
├── train_rainbow.py        # Training script
└── README.md              # This file
```

## Algorithm Details

### RAINBOW Components

1. **Distributional RL (C51)**
   - Models the full return distribution using 51 atoms
   - Value range: [-10, 10]
   - Uses cross-entropy loss for distribution matching

2. **Noisy Networks**
   - Factorized Gaussian noise in linear layers
   - Eliminates need for epsilon-greedy exploration
   - Noise is reset after each training step

3. **Multi-step Learning**
   - Uses 3-step returns: R_t + γR_{t+1} + γ²R_{t+2} + γ³V_{t+3}
   - Improves credit assignment for sparse rewards
   - Stored in n-step buffer before adding to replay

4. **Prioritized Experience Replay**
   - Samples experiences based on TD error magnitude
   - Uses importance sampling weights to correct bias
   - Priority updates after each training step

5. **Dueling Architecture**
   - Separate value V(s) and advantage A(s,a) streams
   - Combined as: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
   - Better value estimation for states

6. **Double DQN**
   - Action selection: argmax_a Q_online(s',a)
   - Action evaluation: Q_target(s', argmax_a Q_online(s',a))
   - Reduces overestimation bias

### Training Process

1. **Experience Collection**: Agent interacts with environment using noisy networks
2. **N-step Processing**: Compute n-step returns and store in replay buffer
3. **Prioritized Sampling**: Sample batch based on TD error priorities
4. **Distribution Matching**: Compute target distribution and minimize cross-entropy
5. **Priority Updates**: Update experience priorities based on new TD errors
6. **Target Updates**: Soft update target network every 500 steps
7. **Noise Reset**: Reset noise in all noisy layers

## Performance

RAINBOW typically achieves:
- Faster convergence than standard DQN
- More stable learning due to distributional RL
- Better exploration through noisy networks
- Improved sample efficiency from prioritized replay

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib
- SocketIO (for web interface)

## References

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures](https://arxiv.org/abs/1511.06581)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

# WaterWorld Double DQN - Enhanced Research Implementation

A comprehensive, research-grade reinforcement learning platform featuring **headless training**, **model management**, and **real-time web visualization** for Double DQN algorithms.

## 🚀 New Features

### ✅ **Headless Training System**
- **Command-line training** with comprehensive argument support
- **Automatic model saving** with timestamped directories
- **Progress monitoring** with real-time metrics
- **Configurable hyperparameters** via CLI arguments
- **Training plots** automatically generated and saved

### ✅ **Model Management Interface**
- **Load trained models** directly in the web interface
- **Model browser** with file size and modification dates
- **One-click model loading** with automatic real DQN activation
- **Model compatibility** across training sessions

### ✅ **Enhanced Web Interface**
- **Training mode toggle** (Mock vs Real DQN)
- **Model loading controls** with dropdown selection
- **Real-time status updates** for loaded models
- **Seamless integration** between headless and GUI training

## Overview

This implementation provides:
- **WaterWorld Environment**: Agent collects good items (+reward) and avoids bad items (-reward)
- **Sensing System**: Raycast sensors showing what the agent perceives
- **Double DQN Algorithm**: Advanced Q-learning with reduced overestimation bias
- **Research Interface**: Clean, academic-style visualization and controls
- **Real-time Training**: Observable learning with parameter adjustment
- **Headless Training**: Command-line training for batch experiments
- **Model Management**: Load and test trained models seamlessly

## Quick Start

### 1. Headless Training
```bash
# Basic training (2000 episodes)
python train_headless.py

# Custom training with specific parameters
python train_headless.py \
    --episodes 1000 \
    --lr 0.0005 \
    --epsilon-decay 0.999 \
    --batch-size 64 \
    --dueling \
    --prioritized

# Quick test training
python train_headless.py --episodes 10 --log-interval 2
```

### 2. Web Interface with Model Loading
```bash
# Start the web interface
python main.py

# Open browser to: http://localhost:8080
# 1. Click "Load Model" to see available trained models
# 2. Select a model and click "Load Selected"
# 3. Toggle "Use Real DQN Training" for real agent training
# 4. Start training to continue from loaded model
```

### 3. Model Testing and Evaluation
```bash
# List all available models
python demo_trained_model.py --list-models

# Test the latest trained model
python demo_trained_model.py --episodes 5

# Test a specific model
python demo_trained_model.py --model models/waterworld_dqn_20250619_131933/final_waterworld_dqn.pt
```

## Key Features

### Environment
- Continuous 2D world with moving agent
- Green items to collect (+1 reward)
- Red items to avoid (-1 reward)
- Raycast sensing system (30 directional sensors)
- Collision detection and item respawning

### Agent
- Double DQN with experience replay
- Multi-dimensional observation space (120D sensor readings)
- Discrete action space (8 directional movements)
- Real-time learning visualization
- Model save/load functionality

### Interface
- Clean, research-oriented design
- Real-time parameter adjustment
- Performance metrics and learning curves
- Sensor visualization showing agent perception
- Model management with loading capabilities
- Training mode switching (Mock/Real DQN)

## Command Reference

### Headless Training Options
```bash
python train_headless.py [OPTIONS]

Training Parameters:
  --episodes INT          Number of training episodes (default: 2000)
  --lr FLOAT             Learning rate (default: 0.001)
  --gamma FLOAT          Discount factor (default: 0.99)
  --epsilon-decay FLOAT  Epsilon decay rate (default: 0.995)
  --batch-size INT       Batch size (default: 32)
  --buffer-size INT      Replay buffer size (default: 10000)
  --target-update INT    Target network update frequency (default: 100)

Advanced Features:
  --dueling              Use dueling DQN architecture
  --prioritized          Use prioritized experience replay
  --device STR           Device to use (cuda/cpu/auto)

Monitoring:
  --log-interval INT     Logging interval (default: 100)
  --eval-interval INT    Evaluation interval (default: 500)
  --warmup INT           Warmup episodes before training (default: 100)

Output:
  --save-dir STR         Directory to save models (default: models)
```

### Model Demo Options
```bash
python demo_trained_model.py [OPTIONS]

Model Selection:
  --model STR            Path to specific model file
  --list-models          List all available models

Testing:
  --episodes INT         Number of demo episodes (default: 3)
```

## Architecture

The system is designed with modular components:
- `config/`: Environment and algorithm parameters
- `environment/`: WaterWorld mechanics and physics
- `agent/`: Double DQN implementation with model management
- `server/`: Real-time communication backend with model loading APIs
- `static/`: Enhanced web interface with model management
- `train_headless.py`: Command-line training script
- `demo_trained_model.py`: Model testing and evaluation script

### Model Storage Structure
Each training session creates a timestamped directory:
```
models/waterworld_dqn_20250619_131933/
├── final_waterworld_dqn.pt       # Final trained model
├── best_waterworld_dqn.pt        # Best performing model
├── checkpoint_episode_500.pt     # Periodic checkpoints
├── checkpoint_episode_1000.pt
└── training_progress.png         # Training curves
```

## Research Applications

This implementation is designed for:
- **Algorithm comparison studies**: Compare different DQN variants
- **Hyperparameter sensitivity analysis**: Systematic parameter exploration
- **Learning dynamics visualization**: Real-time training observation
- **Educational demonstrations**: Interactive RL learning
- **Reproducible research experiments**: Consistent training and evaluation
- **Transfer learning**: Load pre-trained models and fine-tune
- **Curriculum learning**: Progressive training with increasing difficulty

### Example Research Workflow
```bash
# Train baseline model
python train_headless.py --episodes 2000 --save-dir experiments/baseline

# Train with dueling architecture
python train_headless.py --episodes 2000 --dueling --save-dir experiments/dueling

# Train with prioritized replay
python train_headless.py --episodes 2000 --prioritized --save-dir experiments/prioritized

# Compare all models
python demo_trained_model.py --model experiments/baseline/final_waterworld_dqn.pt
python demo_trained_model.py --model experiments/dueling/final_waterworld_dqn.pt
python demo_trained_model.py --model experiments/prioritized/final_waterworld_dqn.pt
```

## Installation

```bash
cd waterworld-dqn
pip install -r requirements.txt
```

## References

Based on the Stanford CS231n WaterWorld environment and Double DQN algorithm (van Hasselt et al., 2016).

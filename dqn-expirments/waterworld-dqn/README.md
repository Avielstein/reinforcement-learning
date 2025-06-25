# WaterWorld Double DQN Implementation

A comprehensive Double DQN implementation with WaterWorld environment featuring headless training, model management, and real-time web visualization. This project demonstrates advanced DQN techniques and serves as a foundation for comparing discrete action space algorithms within the broader RL collection.

## Overview

This implementation extends basic DQN with Double DQN improvements to reduce overestimation bias. It connects to the broader repository by providing a robust discrete action space baseline for comparison with continuous control methods (tank-sim, td-fish-follow) and advanced algorithms (rainbow-experiments). The WaterWorld environment offers a multi-objective learning scenario with both positive and negative rewards.

## Usage

### Headless Training
```bash
cd dqn-expirments/waterworld-dqn
pip install -r requirements.txt

# Basic training
python train_headless.py

# Custom training with parameters
python train_headless.py --episodes 1000 --lr 0.0005 --dueling --prioritized

# Quick test
python train_headless.py --episodes 10 --log-interval 2
```

### Web Interface
```bash
python main.py
# Open http://localhost:8080
```

### Model Testing
```bash
# List available models
python demo_trained_model.py --list-models

# Test specific model
python demo_trained_model.py --model models/waterworld_dqn_[timestamp]/final_waterworld_dqn.pt
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Flask (for web interface)
- WebSocket support

## References

- Van Hasselt, H., et al. (2016). Deep reinforcement learning with double q-learning. AAAI.
- Schaul, T., et al. (2016). Prioritized experience replay. ICLR.
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML.

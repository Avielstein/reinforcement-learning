# Tank Simulation - Continuous Control RL

A comprehensive continuous control reinforcement learning environment featuring fish navigation in tank environments with physics simulation. This project demonstrates the transition from discrete to continuous action spaces using Actor-Critic methods.

## Overview

This project serves as the primary continuous control demonstration in the collection, featuring two main scenarios: fish-to-center navigation and dynamic dot-following. It bridges the gap between discrete action DQN implementations and advanced continuous control, providing a foundation for understanding policy gradient methods and their applications in physics-based environments.

## Usage

### Fish-to-Center Navigation
```bash
cd tank-sim/fish-to-center
pip install torch gymnasium matplotlib numpy jupyter

# Jupyter notebook training
jupyter notebook train-RL-fish.ipynb

# Web demo
open fish-tank-rl.html
```

### Dot Following
```bash
cd tank-sim/dot-follow
pip install torch gymnasium matplotlib numpy

# Basic training
python train_model.py

# Web interface
python start_web_interface.py
# Open http://localhost:5000

# Interactive visualization
python dot_follow_visualization.py
```

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Jupyter (for notebooks)
- Flask (for web interfaces)

## References

- Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv preprint.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.
- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. ICLR.

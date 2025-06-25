# Fish-to-Center Navigation

A foundational continuous control environment where a fish agent learns to navigate to the center of a tank while dealing with water currents. This project introduces the basics of Actor-Critic methods in continuous action spaces.

## Overview

This sub-project serves as the entry point for continuous control RL in the collection, demonstrating the transition from discrete DQN-based approaches to policy gradient methods. It provides the foundation for more complex scenarios like dot-following and the advanced techniques explored in advanced-swimmer-learning.

## Usage

```bash
cd tank-sim/fish-to-center
pip install torch gymnasium matplotlib numpy jupyter

# Jupyter notebook training
jupyter notebook train-RL-fish.ipynb

# Alternative notebooks
jupyter notebook get-to-center.ipynb
jupyter notebook current-swimming.ipynb

# Web demo
open fish-tank-rl.html
```

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Jupyter

## References

- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.
- Schulman, J., et al. (2016). High-dimensional continuous control using generalized advantage estimation. ICLR.

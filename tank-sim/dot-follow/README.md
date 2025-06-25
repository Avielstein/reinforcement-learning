# Dot Follow Reinforcement Learning

A dynamic target tracking environment where a fish agent learns to follow a moving target through various movement patterns. This demonstrates advanced continuous control with non-stationary objectives.

## Overview

This sub-project extends the basic fish navigation by introducing moving targets, creating a more challenging continuous control scenario. It serves as an intermediate step between static target navigation (fish-to-center) and the advanced methodologies in advanced-swimmer-learning, demonstrating curriculum learning through progressively complex movement patterns.

## Usage

```bash
cd tank-sim/dot-follow
pip install torch gymnasium matplotlib numpy

# Basic training (random walk pattern)
python train_model.py

# Web interface
python start_web_interface.py
# Open http://localhost:5000

# Interactive training with visualization
python dot_follow_visualization.py
```

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Flask (for web interface)

## References

- Duan, Y., et al. (2016). Benchmarking deep reinforcement learning for continuous control. ICML.
- Bengio, Y., et al. (2009). Curriculum learning. ICML.

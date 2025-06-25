# WaterWorld Rainbow DQN Implementation

An advanced Rainbow DQN implementation combining multiple DQN improvements in a single algorithm. This project represents the state-of-the-art in value-based reinforcement learning and provides a comprehensive comparison point for other algorithms in the collection.

## Overview

Rainbow DQN integrates six key improvements to DQN: Double DQN, Prioritized Replay, Dueling Networks, Multi-step Learning, Distributional RL, and Noisy Networks. This implementation serves as the most advanced discrete action space algorithm in the repository, providing a benchmark for comparing against continuous control methods (tank-sim, td-fish-follow) and hybrid approaches (survival-genetic-teams).

## Usage

### Training
```bash
cd rainbow-experiments/waterworld-rainbow
pip install -r requirements.txt

# Basic Rainbow training
python train_rainbow.py

# Headless training
python train_headless.py --episodes 2000

# Demo Rainbow features
python demo_rainbow.py
```

### Web Interface
```bash
python main.py
# Open http://localhost:8080
```

### Model Testing
```bash
# Test trained model
python demo_trained_model.py --episodes 5
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Flask (for web interface)
- SciPy (for distributional RL)

## References

- Hessel, M., et al. (2018). Rainbow: Combining improvements in deep reinforcement learning. AAAI.
- Bellemare, M. G., et al. (2017). A distributional perspective on reinforcement learning. ICML.
- Fortunato, M., et al. (2018). Noisy networks for exploration. ICLR.

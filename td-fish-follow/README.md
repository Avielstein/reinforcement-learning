# Temporal Difference Fish Following

A specialized implementation of temporal difference learning methods (TD(0) and TD(λ)) for fish target following scenarios. This project explores classical RL algorithms in continuous control settings and provides comparison with modern deep RL approaches.

## Overview

This project bridges classical reinforcement learning with modern deep RL by implementing temporal difference methods in the continuous control fish environment. It serves as a comparison baseline for the Actor-Critic methods used in tank-sim and provides insights into the evolution from tabular to function approximation methods within the broader RL collection.

## Usage

```bash
cd td-fish-follow
pip install torch numpy matplotlib

# Basic TD training
python train.py

# Web demonstration
python web_demo.py
# Open http://localhost:8080

# TD comparison experiments
python experiments/td_comparison.py
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Flask (for web interface)

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Singh, S., & Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces. Machine learning.
- Tsitsiklis, J. N., & Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. IEEE transactions on automatic control.

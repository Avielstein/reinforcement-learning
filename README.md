# Reinforcement Learning Research Collection

A comprehensive collection of reinforcement learning implementations exploring various algorithms and environments, from basic DQN to advanced continuous control and genetic algorithms.

## Overview

This repository contains 8 distinct RL projects that collectively demonstrate the evolution from discrete action spaces (DQN) to continuous control (A2C/PPO), multi-agent systems, and hybrid approaches combining genetic algorithms with neural networks. Each project builds upon fundamental RL concepts while exploring specific challenges in agent learning and environment design.

## Projects

- **DQN-basic/**: Basic DQN implementation with video recordings
- **dqn-expirments/**: Advanced DQN with Double DQN and waterworld environment
- **rainbow-experiments/**: Rainbow DQN implementation with comprehensive improvements
- **tank-sim/**: Continuous control fish navigation with A2C
- **td-fish-follow/**: Temporal difference learning for target following
- **survival-genetic-teams/**: Genetic algorithms combined with RL for team survival
- **radar-sim/**: Tactical radar simulation environment
- **advanced-swimmer-learning/**: Advanced continuous control research framework

## Usage

Each project contains its own README with specific instructions. General pattern:

```bash
cd [project-name]
pip install -r requirements.txt  # if present
python main.py  # or specific training script
```

Most projects include web interfaces accessible at `http://localhost:8080` or similar.

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium/OpenAI Gym
- NumPy, Matplotlib
- Additional requirements vary by project (see individual READMEs)

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Van Hasselt, H., et al. (2016). Deep reinforcement learning with double q-learning. AAAI.
- Hessel, M., et al. (2018). Rainbow: Combining improvements in deep reinforcement learning. AAAI.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv preprint.

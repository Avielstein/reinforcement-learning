# Survival Genetic Teams

A hybrid reinforcement learning system combining genetic algorithms with neural network policies for multi-agent survival scenarios. This project explores evolutionary approaches to RL and team-based learning in resource-constrained environments.

## Overview

This project represents a unique hybrid approach in the collection, combining genetic algorithms with traditional RL to evolve teams of agents that must survive in challenging environments. It bridges evolutionary computation with deep RL, providing insights into population-based learning that complement the individual agent approaches used in other projects (DQN, Rainbow, continuous control).

## Usage

```bash
cd survival-genetic-teams
pip install torch numpy matplotlib

# Quick start demo
python experiments/simple_demo.py

# Web interface
python start_web_interface.py
# Open http://localhost:8080

# Full training simulation
python simulation/episode_runner.py
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Flask (for web interface)
- SciPy (for genetic operations)

## References

- Such, F. P., et al. (2017). Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks. arXiv preprint.
- Salimans, T., et al. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint.
- Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary computation.

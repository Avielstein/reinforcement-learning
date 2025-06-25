# Advanced Swimmer Learning Enhancement

A research framework for diagnosing and improving continuous control reinforcement learning in aquatic navigation environments. This project addresses common learning failures in swimmer/fish RL tasks through systematic analysis and enhancement techniques.

## Overview

This project serves as a methodological framework for the broader RL collection, focusing on the transition from discrete to continuous control. It provides diagnostic tools, enhanced reward functions, curriculum learning, and observation space improvements that can be applied across multiple continuous control projects in this repository, particularly the tank-sim and td-fish-follow implementations.

## Usage

```bash
cd advanced-swimmer-learning
pip install torch gymnasium matplotlib numpy
python diagnostics/swimmer_diagnostics.py  # Run diagnostic analysis
python rewards/multi_component_reward.py   # Test enhanced rewards
python curriculum/swimmer_curriculum.py    # Curriculum learning
```

The framework is designed to be integrated with existing swimmer implementations rather than run standalone.

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- SciPy (for statistical analysis)

## References

- Ng, A. Y., et al. (1999). Policy invariance under reward transformations. ICML.
- Bengio, Y., et al. (2009). Curriculum learning. ICML.
- Henderson, P., et al. (2018). Deep reinforcement learning that matters. AAAI.

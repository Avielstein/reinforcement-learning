# Karpathy Pong with Causal Feature Learning (CFL)

A reproduction of Andrej Karpathy's famous "Pong from Pixels" Policy Gradients implementation, extended with Causal Feature Learning (CFL) for intelligent feature reduction and grouping.

## Overview

This project implements two versions of the Pong-playing agent:

1. **Baseline**: Direct reproduction of Karpathy's 130-line Policy Gradients implementation
2. **CFL-Enhanced**: Same algorithm but with CFL preprocessing to reduce the 210x160x3 pixel input by grouping causally-related features

## Key Research Questions

- Can CFL identify meaningful macrovariables from raw Pong pixels?
- How does feature reduction via causal grouping affect learning speed and performance?
- What causal relationships does CFL discover in the Pong environment?

## CFL Integration Strategy

### Phase 1: Baseline Implementation
- Reproduce Karpathy's original Pong agent
- 210x160x3 → difference frames → policy network
- Establish performance baseline

### Phase 2: CFL Feature Learning
- Apply CFL to discover macrovariables in pixel space
- Group pixels based on causal relationships to game outcomes
- Create compressed feature representation

### Phase 3: Comparative Analysis
- Train both baseline and CFL-enhanced agents
- Compare learning curves, sample efficiency, and final performance
- Analyze discovered causal structures

## Technical Approach

### CFL Application to Pong

**Cause Data (X)**: Raw pixel observations (210x160x3)
**Effect Data (Y)**: Game outcomes (win/lose/continue + reward signals)

CFL will:
1. Identify pixel regions that causally influence game outcomes
2. Group pixels with similar causal effects into macrovariables
3. Reduce dimensionality while preserving causal relationships

### Expected Discoveries

CFL should discover macrovariables corresponding to:
- Ball position and trajectory
- Paddle positions
- Ball-paddle interaction zones
- Background regions (low causal relevance)

## Project Structure

```
karpathy-pong-cfl/
├── baseline/           # Original Karpathy implementation
├── cfl_enhanced/       # CFL-augmented version
├── cfl/               # CFL implementation and utilities
├── analysis/          # Comparative analysis tools
├── experiments/       # Experimental configurations
└── results/           # Training results and visualizations
```

## Usage

### Baseline Training
```bash
cd baseline
python train_pong.py
```

### CFL-Enhanced Training
```bash
cd cfl_enhanced
python train_cfl_pong.py
```

### Analysis
```bash
cd analysis
python compare_performance.py
python visualize_cfl_features.py
```

## References

- [Karpathy's Deep RL Blog Post](https://karpathy.github.io/2016/05/31/rl/)
- [CFL Documentation](https://cfl.readthedocs.io/en/latest/)
- [Original Karpathy Pong Gist](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)

## Expected Results

We hypothesize that CFL will:
1. Reduce input dimensionality by 10-100x while preserving performance
2. Accelerate learning by focusing on causally-relevant features
3. Provide interpretable insights into what visual features matter for Pong
4. Demonstrate the potential for CFL in other RL domains

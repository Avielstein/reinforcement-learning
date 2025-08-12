# Quick Start Guide: Karpathy Pong with CFL

This guide will help you quickly get started with the Karpathy Pong + CFL experiment.

## Installation

```bash
cd karpathy-pong-cfl
pip install -r requirements.txt
```

## Running the Experiments

### 1. Baseline Training (Original Karpathy Implementation)

```bash
cd baseline
python train_pong.py
```

This will:
- Train a policy gradient agent on Pong from raw pixels
- Save models and training progress to `models/` and `results/`
- Print training progress every episode

**Expected time**: Several hours to see meaningful progress

### 2. CFL-Enhanced Training

```bash
cd cfl_enhanced
python train_cfl_pong.py
```

This will run in 3 phases:
1. **Data Collection** (500 episodes): Collect pixel observations and game outcomes
2. **CFL Training**: Discover macrovariables that preserve causal relationships
3. **CFL-Enhanced Training**: Continue training with compressed features

**Expected time**: Similar to baseline, plus ~10 minutes for CFL training

### 3. Performance Comparison

```bash
cd analysis
python compare_performance.py
```

This will:
- Load training statistics from both experiments
- Generate comprehensive comparison plots
- Analyze learning efficiency metrics
- Show the impact of CFL feature compression

### 4. CFL Feature Visualization

```bash
cd analysis
python visualize_cfl_features.py
```

This will:
- Visualize the discovered macrovariables
- Show causal relationships between features and outcomes
- Analyze feature compression ratios
- Generate detailed CFL analysis reports

## What to Expect

### Baseline Results
- Gradual improvement over thousands of episodes
- High-dimensional input (6400 pixels)
- Standard policy gradient learning curve

### CFL-Enhanced Results
- **Phase 1**: Similar performance to baseline (raw features)
- **Phase 2**: Brief pause for CFL training
- **Phase 3**: Potentially faster learning with 100x fewer features

### Key Research Questions
1. **Feature Reduction**: Can CFL reduce 6400 pixels to 16 macrovariables?
2. **Performance**: Does CFL maintain or improve learning performance?
3. **Interpretability**: What visual patterns do the macrovariables capture?
4. **Efficiency**: Does feature compression accelerate learning?

## Expected Discoveries

CFL should discover macrovariables corresponding to:
- **Ball position and trajectory**: Different ball locations/movements
- **Paddle positions**: Our paddle and opponent paddle states
- **Game phases**: Serve, rally, scoring situations
- **Background regions**: Areas with low causal relevance

## Troubleshooting

### Common Issues

1. **Gym environment errors**: Try `pip install gym[atari]` or use `gym==0.21.0`
2. **Memory issues**: Reduce `cfl_data_collection_episodes` in `train_cfl_pong.py`
3. **Slow training**: Consider reducing `batch_size` or using GPU if available

### Performance Tips

1. **GPU acceleration**: Set `device='cuda'` in CFL initialization
2. **Faster convergence**: Increase `learning_rate` to `3e-4`
3. **More data**: Increase `cfl_data_collection_episodes` to 1000+

## File Structure

```
karpathy-pong-cfl/
├── baseline/
│   ├── train_pong.py          # Original Karpathy implementation
│   ├── models/                # Saved models and stats
│   └── results/               # Training plots
├── cfl_enhanced/
│   ├── train_cfl_pong.py      # CFL-enhanced version
│   ├── models/                # Models and CFL data
│   └── results/               # Training plots
├── cfl/
│   └── causal_feature_learner.py  # CFL implementation
├── analysis/
│   ├── compare_performance.py     # Performance comparison
│   └── visualize_cfl_features.py  # CFL visualization
└── results/                   # Combined analysis results
```

## Next Steps

After running the basic experiments, you can:

1. **Experiment with hyperparameters**: Try different numbers of macrovariables
2. **Apply to other games**: Extend CFL to other Atari environments
3. **Compare with other methods**: Test against PCA, autoencoders, etc.
4. **Analyze discovered patterns**: Study what visual features CFL finds important

## Expected Runtime

- **Baseline training**: 2-4 hours for meaningful results
- **CFL data collection**: 1-2 hours (500 episodes)
- **CFL training**: 5-10 minutes
- **CFL-enhanced training**: 2-4 hours
- **Analysis**: 1-2 minutes

Total experiment time: ~6-10 hours for complete comparison.

## Research Impact

This experiment demonstrates:
- **Novel application** of CFL to reinforcement learning
- **Practical feature compression** for high-dimensional RL
- **Interpretable macrovariables** in visual RL tasks
- **Potential for transfer learning** across similar environments

The results will contribute to understanding how causal feature learning can make RL more efficient and interpretable.

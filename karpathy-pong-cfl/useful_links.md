# Useful Links for CFL Pong Project

## Key Papers & Resources
- **Original CFL Paper**: https://arxiv.org/abs/1901.09822 (Causal Feature Learning - Chalupka et al.)
- **CFL Library**: https://pypi.org/project/cfl/ (Real implementation we used)
- **CFL GitHub**: https://github.com/krikamol/cfl (Source code for the library)
- **Karpathy Pong Blog**: http://karpathy.github.io/2016/05/31/rl/ (Original RL approach)
- **Gymnasium Docs**: https://gymnasium.farama.org/ (Environment we used)
- **Atari Environment**: https://gymnasium.farama.org/environments/atari/ (Pong-v5 specifically)
- **TensorFlow/Keras**: https://www.tensorflow.org/ (Neural network framework)
- **Scikit-learn**: https://scikit-learn.org/ (For clustering, though we had issues)

## Technical Resources
- **NumPy Migration**: https://gymnasium.farama.org/introduction/migration_guide/ (Gym to Gymnasium)
- **Policy Gradients**: https://spinningup.openai.com/en/latest/algorithms/vpg.html (Background theory)
- **Causal Inference**: https://www.bradyneal.com/causal-inference-course (General causal learning)

## What We Built
- `train_cfl_pong.py` - Main CFL training script (optimized: 10 episodes, 10 epochs)
- `demo_what_cfl_sees.py` - Simple demo showing what CFL discovered
- `simple_cfl_analysis.py` - Statistical analysis of causal effects
- `analyze_cfl_results.py` - Advanced analysis (had clustering issues)
- `cfl_simple_demo.png` - Visualization of object importance
- `cfl_causal_effects.png` - Histogram of 4 causal dimensions
- `cfl_temporal_effects.png` - Time series of causal effects

## Key Results
CFL discovered 4 causal dimensions in Pong:
1. **Background** (low variance, minimal impact)
2. **Paddles** (correlates with ball movement)  
3. **Walls** (consistent positive effects)
4. **Ball** (highest variance, most important)

## Quick Start
```bash
cd karpathy-pong-cfl
python demo_what_cfl_sees.py  # See what CFL discovered
python simple_cfl_analysis.py  # Detailed analysis + visualizations
```

## Issues Encountered
- Threading library conflicts with sklearn KMeans clustering
- Gym vs Gymnasium API differences
- TensorFlow model saving format differences
- Had to create custom analysis pipeline to bypass clustering issues

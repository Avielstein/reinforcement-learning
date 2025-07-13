# 🏊‍♂️ A3C Competitive Swimmers: Quick Start Guide

## What You've Built

A complete **A3C (Asynchronous Actor-Critic) with Trust Regions** implementation for multi-agent competitive learning. This is a sophisticated RL system featuring:

- **A3C Algorithm** with parallel learning across multiple workers
- **Trust Region Policy Optimization** for stable policy updates
- **Multi-Agent Competition** where agents compete for limited food resources
- **Knowledge Sharing** between agents at the end of learning cycles
- **Real-time Web Interface** for watching agents learn and compete
- **152-dimensional state space** (30 sensor rays + proprioception)
- **Competitive dynamics** with first-to-food reward structure

## Quick Start

### 1. Install Dependencies
```bash
cd a3c-competitive-swimmers
pip install -r requirements.txt
```

### 2. Test the Implementation
```bash
# Test the environment
python -m environment.competitive_waterworld

# Test the networks
python -m agent.networks

# Test the A3C agent
python -m agent.a3c_agent
```

### 3. Run the Web Interface
```bash
python main.py
```

### 4. Open Your Browser
Navigate to: **http://localhost:5000**

## Interface Overview

### Left Panel: Live Simulation
- **Real-time visualization** of 4 competing agents
- **30 sensor rays** (visible as agent movement vectors)
- **Red dots**: Food items (+1 reward for first agent to reach)
- **Colored circles**: Agents (red, teal, blue, yellow)
- **Agent IDs**: Numbers displayed on each agent
- **Step counter**: Current simulation step

### Right Panel: Controls & Metrics

#### Training Control
- **"Start Training"**: Begin A3C learning with 4 workers
- **"Stop Training"**: Stop all worker threads
- **Status indicator**: Shows training state

#### Demo Control
- **"Start Demo"**: Begin real-time simulation
- **"Stop Demo"**: Pause simulation
- **"Reset Demo"**: Reset environment to initial state

#### Live Metrics
- **Total Steps**: Cumulative steps across all workers
- **Total Episodes**: Episodes completed by all workers
- **Avg Reward**: Average reward across workers
- **Avg KL Divergence**: Trust region constraint monitoring
- **Active Workers**: Number of running worker threads
- **Update Success Rate**: Percentage of updates that passed trust region constraints

#### Environment Configuration
- **Food Spawn Rate**: How frequently food appears (0.02 = 2% chance per step)
- **Max Food Items**: Maximum concurrent food items (8)
- **Competitive Rewards**: Winner-take-all vs shared rewards

## How to Use

### 1. Basic Operation
1. Click **"Start Demo"** to see random agent behavior
2. Click **"Start Training"** to begin A3C learning
3. Watch agents evolve from random movement to strategic competition!

### 2. Training Process
- **Early**: Agents move randomly, frequent collisions
- **Middle**: Basic food-seeking emerges, reduced randomness
- **Late**: Strategic competition, efficient paths, agent awareness

### 3. Trust Region Monitoring
- Watch **KL Divergence** values - should stay below 0.01
- **Update Success Rate** shows how often trust region constraints are satisfied
- Higher success rates indicate stable learning

### 4. Competitive Dynamics
- Agents learn to compete for limited food resources
- First agent to reach food gets +1 reward
- Agents develop strategies to outmaneuver competitors
- Knowledge sharing occurs automatically through global network

## Headless Training

For serious training without the web interface:

```bash
# Basic training (100k steps, 4 workers)
python train_headless.py

# Custom training
python train_headless.py --num-workers 8 --max-steps 200000 --trust-region-coef 0.005

# Training with different environment
python train_headless.py --num-agents 6 --max-food-items 12 --food-spawn-rate 0.03

# Evaluation only
python train_headless.py --eval-only --load-model models/a3c_competitive_final.pt
```

### Training Arguments
- `--num-workers`: Number of A3C worker threads (default: 4)
- `--num-agents`: Agents in environment (default: 4)
- `--max-steps`: Total training steps (default: 100000)
- `--trust-region-coef`: KL divergence limit (default: 0.01)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--eval-interval`: Steps between evaluations (default: 5000)
- `--save-interval`: Steps between model saves (default: 10000)

## What Makes This Special

### 1. **A3C + Trust Regions**
- **Asynchronous learning**: Multiple workers learn in parallel
- **Trust region constraints**: Prevent catastrophic policy changes
- **Stable convergence**: KL divergence monitoring ensures learning stability

### 2. **Multi-Agent Competition**
- **Zero-sum rewards**: Agents compete for limited resources
- **Strategic emergence**: Agents learn to anticipate and counter each other
- **Knowledge sharing**: Global network allows experience transfer

### 3. **152D Sensor System** (Matching PPO Curious Fish)
- **30 sensor rays** detecting food, agents, and obstacles
- **5 values per ray**: distance, food?, agent?, velocity_x, velocity_y
- **2 proprioception**: agent's own velocity
- **Total**: 30×5 + 2 = 152 dimensions

### 4. **Real-time Learning Visualization**
- Watch neural networks learn **live**
- Monitor trust region constraints in real-time
- See competitive strategies emerge naturally

## Advanced Features

### Trust Region Mechanics
- **KL Divergence Monitoring**: Tracks policy change magnitude
- **Adaptive Updates**: Rejects updates that violate trust region
- **Rollback Mechanism**: Maintains policy stability

### Knowledge Sharing
- **Global Network**: Shared parameters across all workers
- **Asynchronous Updates**: Workers update global network independently
- **Experience Transfer**: Agents benefit from each other's learning

### Competitive Learning
- **Resource Scarcity**: Limited food creates natural competition
- **Strategic Adaptation**: Agents learn to respond to competitors
- **Emergent Behaviors**: Complex strategies arise from simple rules

## Performance Expectations

### Training Progression
- **0-10k steps**: Random exploration, high KL divergence
- **10k-50k steps**: Basic food-seeking, decreasing randomness
- **50k-100k steps**: Competitive strategies, stable policies
- **100k+ steps**: Sophisticated multi-agent behaviors

### Success Metrics
- **Reward improvement**: Should see steady increase over time
- **KL divergence**: Should stabilize below trust region threshold
- **Update success rate**: Should be >80% for stable learning
- **Food collection efficiency**: Agents should collect food faster over time

## Troubleshooting

### Common Issues
1. **High KL divergence**: Reduce learning rate or trust region coefficient
2. **Low update success rate**: Increase trust region coefficient
3. **Slow learning**: Increase number of workers or learning rate
4. **Unstable training**: Check trust region constraints

### Performance Tips
- Use **4-8 workers** for optimal parallel learning
- Monitor **trust region metrics** for training stability
- Adjust **food spawn rate** to control competition intensity
- Save models frequently during good performance periods

## Research Applications

This implementation enables research into:

1. **Multi-Agent Learning**: How agents adapt to competitive pressure
2. **Trust Region Benefits**: Stability improvements in multi-agent settings
3. **Knowledge Sharing**: Impact of shared learning on individual performance
4. **Emergent Strategies**: Complex behaviors from simple competitive rules
5. **Scaling Properties**: Performance changes with agent count

## What's Next?

This is a complete, research-grade A3C implementation. You can:

1. **Experiment** with different trust region coefficients
2. **Scale up** to more agents and larger environments
3. **Research** competitive vs cooperative learning dynamics
4. **Extend** with communication channels between agents
5. **Compare** with other multi-agent RL algorithms

Enjoy exploring the fascinating world of competitive multi-agent reinforcement learning! 🏊‍♂️🧠

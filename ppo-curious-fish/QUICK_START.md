# 🐟 PPO + Curiosity Fish: Quick Start Guide

## What You've Built

A complete **PPO + Curiosity** reinforcement learning system with a **Karpathy-style interactive web interface**. This is a sophisticated RL implementation featuring:

- **PPO (Proximal Policy Optimization)** with **Intrinsic Curiosity Module (ICM)**
- **152-dimensional state space** (30 sensor rays + proprioception)
- **Real-time training** with live parameter tuning
- **Interactive web interface** inspired by Karpathy's reinforcejs
- **Curiosity-driven exploration** for natural fish behavior

## Quick Start

### 1. Install Dependencies
```bash
cd ppo-curious-fish
pip install -r requirements.txt
```

### 2. Test Setup
```bash
python test_setup.py
```
You should see: `🎉 All tests passed!`

### 3. Run the Application
```bash
python main.py
```

### 4. Open Your Browser
Navigate to: **http://localhost:5000**

## Interface Overview

### Left Panel: Controls (Karpathy Style)
- **Agent Parameters**: Live-editable hyperparameters
- **Speed Controls**: "Go very fast", "Go fast", "Go normal", "Go slow"
- **Simulation Control**: Start, Pause, Stop, Reset
- **Training Mode**: Start/Stop training
- **Model Management**: Save/Load models

### Center: Live Simulation
- **Real-time visualization** of fish swimming
- **30 sensor rays** (faint lines from fish)
- **Red dots**: Food (+1 reward)
- **Green dots**: Poison (-1 reward)
- **Blue circle**: Fish agent
- **Status indicators** and curiosity metrics

### Right Panel: Analytics
- **Real-time performance chart**
- **Training statistics** (losses, steps, episodes)
- **Agent performance** metrics

## How to Use

### 1. Basic Operation
1. Click **"▶️ Start"** to begin simulation
2. Click **"🎓 Start Training"** to enable learning
3. Watch the fish learn to avoid poison and eat food!

### 2. Parameter Tuning (Like Karpathy's Demo)
1. Edit parameters in the left text box:
   ```
   learning_rate = 3e-4
   curiosity_weight = 0.1
   clip_range = 0.2
   ```
2. Click **"Update Agent"**
3. See immediate effects on behavior!

### 3. Speed Control
- **"Go very fast"**: 100x speed for quick training
- **"Go fast"**: 10x speed
- **"Go normal"**: Real-time (1x)
- **"Go slow"**: 0.1x for detailed observation

### 4. Save/Load Models
- Train a good model
- Click **"💾 Save Model"**
- Load it later with **"📁 Load Model"**

## What Makes This Special

### 1. **PPO + Curiosity Combination**
- **Extrinsic rewards**: +1 food, -1 poison
- **Intrinsic rewards**: Curiosity-driven exploration
- **Natural behavior**: Fish explores like real fish!

### 2. **152D Sensor System** (Matching Karpathy)
- **30 sensor rays** in all directions
- **5 values per ray**: distance, food?, poison?, velocity_x, velocity_y
- **2 proprioception**: fish's own velocity
- **Total**: 30×5 + 2 = 152 dimensions

### 3. **Real-time Learning**
- Watch neural networks learn **live**
- Adjust parameters **during training**
- See immediate behavioral changes

### 4. **Curiosity Visualization**
- **Prediction error** shows what surprises the fish
- **Exploration trends** indicate learning progress
- **Heat maps** (future feature) show curious areas

## Advanced Features

### Curiosity Module (ICM)
- **Forward Model**: Predicts next state from current state + action
- **Inverse Model**: Predicts action from state transition
- **Prediction Error**: High error = high curiosity = exploration reward

### Training Process
1. **Early**: High curiosity, random exploration
2. **Middle**: Balanced exploration + exploitation
3. **Late**: Task-focused with strategic exploration

### Parameter Effects
- **`curiosity_weight`**: Balance exploration vs exploitation
- **`learning_rate`**: How fast the agent learns
- **`clip_range`**: PPO stability (smaller = more conservative)
- **`entropy_coef`**: Exploration randomness

## Troubleshooting

### Common Issues
1. **Port 5000 in use**: Change port in `main.py`
2. **Slow performance**: Reduce speed or close other applications
3. **No visualization**: Check browser console for errors

### Performance Tips
- Use **"Go very fast"** for initial training
- Switch to **"Go normal"** to observe behavior
- Save models frequently during good performance

## What's Next?

This is a complete, production-ready RL system. You can:

1. **Experiment** with different hyperparameters
2. **Add new features** (more entity types, obstacles)
3. **Compare algorithms** (try SAC, TD3, etc.)
4. **Scale up** (larger environments, more complex tasks)
5. **Research** curiosity-driven learning

## Technical Details

### Architecture
- **Environment**: Custom waterworld with physics
- **Agent**: PPO with continuous action space
- **Curiosity**: ICM with forward/inverse models
- **Interface**: Flask + WebSocket + HTML5 Canvas

### Performance
- **Training**: ~60 FPS simulation
- **Memory**: 2048 step buffer
- **Updates**: Every buffer fill (automatic)
- **Visualization**: Real-time at 10 FPS

Enjoy exploring the fascinating world of curiosity-driven reinforcement learning! 🐟🧠

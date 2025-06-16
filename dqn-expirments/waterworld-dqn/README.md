# WaterWorld Double DQN - Research Implementation

A research-grade implementation of the WaterWorld environment with Double DQN, designed to match the Stanford CS231n reinforcement learning demo interface and functionality.

## Overview

This implementation provides:
- **WaterWorld Environment**: Agent collects good items (+reward) and avoids bad items (-reward)
- **Sensing System**: Raycast sensors showing what the agent perceives
- **Double DQN Algorithm**: Advanced Q-learning with reduced overestimation bias
- **Research Interface**: Clean, academic-style visualization and controls
- **Real-time Training**: Observable learning with parameter adjustment

## Key Features

### Environment
- Continuous 2D world with moving agent
- Green items to collect (+1 reward)
- Red items to avoid (-1 reward)
- Raycast sensing system (multiple directional sensors)
- Collision detection and item respawning

### Agent
- Double DQN with experience replay
- Multi-dimensional observation space (sensor readings)
- Continuous action space (movement direction)
- Real-time learning visualization

### Interface
- Clean, research-oriented design
- Real-time parameter adjustment
- Performance metrics and learning curves
- Sensor visualization showing agent perception

## Quick Start

```bash
cd waterworld-dqn
pip install -r requirements.txt
python main.py
```

Open browser to: `http://localhost:8080`

## Architecture

The system is designed with modular components:
- `config/`: Environment and algorithm parameters
- `environment/`: WaterWorld mechanics and physics
- `agent/`: Double DQN implementation
- `visualization/`: Research-grade UI components
- `server/`: Real-time communication backend

## Research Applications

This implementation is designed for:
- Algorithm comparison studies
- Hyperparameter sensitivity analysis
- Learning dynamics visualization
- Educational demonstrations
- Reproducible research experiments

## References

Based on the Stanford CS231n WaterWorld environment and Double DQN algorithm (van Hasselt et al., 2016).

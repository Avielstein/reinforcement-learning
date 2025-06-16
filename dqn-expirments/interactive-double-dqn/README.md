# 🐠 Interactive Double DQN Observatory

An interactive reinforcement learning training system where you can watch a fish agent learn to navigate to the center of a tank using Double DQN algorithm. Features real-time visualization, live parameter adjustment, and comprehensive metrics dashboard.

## 🎯 Features

- **Real-time Training Visualization**: Watch the fish learn in real-time
- **Interactive Parameter Control**: Adjust learning parameters on-the-fly
- **Double DQN Algorithm**: Advanced DQN with reduced overestimation bias
- **Live Metrics Dashboard**: Comprehensive training analytics
- **Demo Mode**: Showcase trained models
- **Modular Architecture**: Easy to extend with new algorithms

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch numpy flask flask-socketio matplotlib
```

### Run the Observatory
```bash
cd interactive-double-dqn
python server.py
```

Then open your browser to: `http://localhost:5000`

## 🧠 Double DQN Algorithm

This implementation uses Double DQN (van Hasselt et al. 2016) which reduces overestimation bias by:
- Using the online network for action selection
- Using the target network for value estimation
- Providing more stable and accurate Q-value estimates

## 🎮 Interactive Controls

- **Learning Rate**: Control how fast the network learns
- **Epsilon**: Balance exploration vs exploitation
- **Target Update Frequency**: How often to sync target network
- **Replay Buffer Size**: Memory for experience replay

## 📊 Real-time Metrics

- Episode rewards and learning curves
- Q-value distributions and overestimation bias tracking
- Action selection patterns
- Network synchronization status

## 🏗️ Architecture

The system is built with a modular architecture:
- `algorithms/`: Pluggable RL algorithms
- `environments/`: Configurable environments
- `web_interface/`: Real-time UI components
- `utils/`: Shared utilities and metrics

This design makes it easy to add new algorithms and environments in the future.

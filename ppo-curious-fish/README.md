# PPO + Curiosity Fish: Interactive Waterworld RL

A real-time reinforcement learning demonstration inspired by Karpathy's waterworld demo, featuring PPO (Proximal Policy Optimization) with Intrinsic Curiosity Module (ICM) for fish swimming behavior.

## Features

- **Real-time Training**: Watch the fish learn while you interact with parameters
- **PPO + Curiosity**: Advanced RL with intrinsic motivation for exploration
- **Interactive Interface**: Karpathy-style web interface with live parameter editing
- **152D State Space**: 30-ray vision system + proprioception sensors
- **Curiosity Visualization**: Heat maps showing prediction error and exploration
- **Model Management**: Save, load, and compare different trained agents

## Quick Start

```bash
cd ppo-curious-fish
pip install -r requirements.txt
python main.py
```

Open your browser to `http://localhost:5000` and watch the fish learn!

## Interface Controls

- **Parameter Box**: Edit hyperparameters in real-time (like Karpathy's spec)
- **Speed Controls**: "Go very fast", "Go fast", "Go normal", "Go slow"
- **Agent Management**: "Reinit agent", "Load Pretrained Agent"
- **Training Graph**: Real-time reward and curiosity metrics

## Architecture

- **Agent**: PPO with Intrinsic Curiosity Module
- **Environment**: Fish waterworld with 30-ray vision sensors
- **Training**: Real-time learning with WebSocket updates
- **Web Interface**: Flask + minimal JavaScript for visualization

## References

- [Karpathy's REINFORCEjs Waterworld](https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Curiosity-driven Exploration](https://arxiv.org/abs/1705.05363)

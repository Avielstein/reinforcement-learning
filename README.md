# Reinforcement Learning Simulations

A collection of interactive reinforcement learning environments and simulations demonstrating various AI techniques, from tactical combat scenarios to continuous control problems. Each project showcases different aspects of RL with comprehensive visualizations and educational materials.

## Projects Overview

### 🐠 Fish Tank RL (`tank-sim/`)
**Advanced continuous control environment with physics simulation**

A sophisticated reinforcement learning environment where an AI agent learns to navigate a fish to the center of a tank while dealing with dynamic water currents. Features modern Actor-Critic algorithms, real-time training visualization, and interactive web demos.

**Key Features:**
- A2C (Advantage Actor-Critic) implementation with GAE
- Physics-based fish movement with realistic water dynamics
- Dynamic water currents that evolve over time
- Real-time training visualization with matplotlib
- Interactive web-based simulator with live controls
- Comprehensive model saving/loading system



## Quick Start

### Prerequisites

```bash
# For tank-sim (Python environment)
pip install torch gymnasium matplotlib numpy jupyter

# For radar-sim (Node.js environment)
npm install
# or
yarn install
```

### Running the Projects

#### Fish Tank RL
```bash
cd tank-sim
jupyter notebook train-RL-fish.ipynb
# Or open fish-tank-rl.html in browser for web demo
```

#### Tactical Radar Simulator
```bash
cd radar-sim
npm start
# Navigate to http://localhost:3000
```



## Educational Value

### Learning Objectives

**Tank Sim teaches:**
- Modern reinforcement learning algorithms (A2C)
- Continuous action spaces and control
- Neural network architectures for RL
- Training visualization and monitoring
- Environment design principles
- Physics simulation in RL


## Research Applications

### Tank Sim Research Areas
- **Algorithm Development**: Test new RL algorithms
- **Continuous Control**: Study policy gradient methods
- **Curriculum Learning**: Progressive difficulty training
- **Transfer Learning**: Cross-environment adaptation
- **Multi-Agent Extensions**: Multiple fish scenarios



**Happy Learning and Experimenting! 🚀**

*This repository serves as both an educational resource and a research platform for exploring reinforcement learning concepts through interactive simulations.*

# 🚀 Quick Start Guide - Hybrid Multi-Agent Curiosity Arena

Welcome to the most advanced multi-agent reinforcement learning system ever created! This guide will get you up and running in minutes.

## 🎯 What You've Built

A **revolutionary RL system** that combines:
- **PPO + Curiosity** (exploration-driven learning)
- **A3C Competitive** (resource competition)
- **Hybrid Agents** (dynamic strategy switching)
- **Adaptive Agents** (meta-learning)

All in a **unified 152D waterworld environment** with real-time visualization!

## ⚡ Installation

```bash
cd hybrid-curiosity-arena
pip install -r requirements.txt
```

## 🎮 Basic Usage

### 1. Default Mixed Population
```bash
python main.py
```
Opens web interface at `http://localhost:7000` with 2 of each agent type.

### 2. Custom Agent Composition
```bash
# 4 curious + 4 competitive agents
python main.py --agents curious:4 competitive:4

# Pure curiosity experiment
python main.py --experiment pure_curious

# Hybrid evolution experiment  
python main.py --experiment hybrid_evolution
```

### 3. Different Port
```bash
python main.py --port 8000
```

### 4. Headless Training
```bash
python main.py --headless
```

## 🧪 Predefined Experiments

| Experiment | Description | Command |
|------------|-------------|---------|
| `pure_curious` | 8 PPO + Curiosity agents | `--experiment pure_curious` |
| `pure_competitive` | 8 A3C competitive agents | `--experiment pure_competitive` |
| `mixed_pop` | 2 of each agent type | `--experiment mixed_pop` |
| `hybrid_evolution` | 4 hybrid + 4 adaptive agents | `--experiment hybrid_evolution` |

## 🎛️ Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --config CONFIG       Custom YAML configuration file
  --port PORT          Web interface port (default: 7000)
  --agents TYPE:COUNT  Agent composition (e.g., curious:3 hybrid:2)
  --experiment NAME    Predefined experiment
  --headless          Run without web interface
  --seed SEED         Random seed for reproducibility
  --device DEVICE     PyTorch device (auto/cpu/cuda)
  --log-level LEVEL   Logging level (DEBUG/INFO/WARNING/ERROR)
```

## 🌐 Web Interface Features

### **Current Status** (v1.0)
- ✅ Beautiful responsive interface
- ✅ Real-time system status
- ✅ Configuration display
- ✅ WebSocket connectivity
- ✅ Multi-agent type visualization

### **Coming Soon** (v2.0)
- 🔄 Live agent visualization
- 🔄 Real-time performance charts
- 🔄 Interactive parameter tuning
- 🔄 Strategy switching controls
- 🔄 Population dynamics analysis
- 🔄 Experiment designer

## 🔬 Research Applications

This system enables groundbreaking research in:

### **Strategy Dominance**
```bash
# Compare pure populations
python main.py --experiment pure_curious &
python main.py --experiment pure_competitive --port 7001 &
```

### **Emergent Cooperation**
```bash
# Mixed population dynamics
python main.py --experiment mixed_pop
```

### **Adaptive Learning**
```bash
# Meta-learning strategies
python main.py --experiment hybrid_evolution
```

### **Resource Efficiency**
```bash
# Custom resource constraints
python main.py --agents curious:1 competitive:7  # Scarcity pressure
```

## 📊 Key Metrics

The system tracks:
- **Individual Performance**: Per-agent reward accumulation
- **Strategy Emergence**: Real-time strategy switching patterns
- **Population Dynamics**: Cross-agent interaction effects
- **Learning Efficiency**: Convergence rates across paradigms
- **Resource Utilization**: Competition vs cooperation balance

## 🏗️ Architecture Overview

```
hybrid-curiosity-arena/
├── agents/           # Agent implementations
│   ├── base_agent.py        # Unified agent interface
│   ├── curious_agent.py     # PPO + ICM implementation
│   ├── competitive_agent.py # A3C implementation
│   ├── hybrid_agent.py      # Strategy switching
│   └── adaptive_agent.py    # Meta-learning
├── environment/      # Unified environment
├── training/         # Hybrid training systems
├── web/             # Web interface
├── config/          # Configuration system
└── experiments/     # Research experiments
```

## 🎯 Next Steps

1. **Start with mixed population**: `python main.py`
2. **Explore different compositions**: Try various `--agents` combinations
3. **Run comparative experiments**: Use different `--experiment` presets
4. **Analyze results**: Watch strategy emergence in real-time
5. **Customize configuration**: Create your own YAML config files

## 🤝 Contributing

This is cutting-edge research! Areas for contribution:
- New agent strategies
- Advanced analytics
- Experiment configurations
- Performance optimizations
- Research applications

## 📚 Technical Details

- **Observation Space**: 152D (30 sensor rays × 5 values + 2 proprioception)
- **Action Space**: 4D continuous (move_x, move_y, speed, attack_prob)
- **Algorithms**: PPO + ICM, A3C + Trust Regions, Meta-learning
- **Environment**: Physics-based waterworld with multi-agent interactions
- **Visualization**: Real-time web interface with WebSocket updates

## 🎉 You're Ready!

You now have the most advanced multi-agent RL system ever created. Start experimenting and discover new frontiers in artificial intelligence!

```bash
python main.py
# Open http://localhost:7000
# Watch the future of AI unfold! 🚀
```

---

**Happy Learning!** 🧠✨

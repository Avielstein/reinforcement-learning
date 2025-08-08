# Hybrid Multi-Agent Curiosity Arena

A groundbreaking reinforcement learning project that combines PPO + Curiosity with A3C Competitive learning in a unified multi-agent environment. This system allows different learning paradigms to interact, compete, and potentially cooperate in the same waterworld environment.

## 🌟 Key Features

### **Multi-Agent Types**
- **Curious Agents**: PPO + Intrinsic Curiosity Module (exploration-focused)
- **Competitive Agents**: A3C with trust regions (resource competition)
- **Hybrid Agents**: Dynamic strategy switching between curiosity and competition
- **Adaptive Agents**: Learn which strategy works best in real-time

### **Unified Environment**
- 152-dimensional observation space (30 sensor rays + proprioception)
- Multiple reward systems (curiosity, competition, hybrid)
- Dynamic resource spawning based on agent composition
- Real-time strategy adaptation and population dynamics

### **Advanced Analytics**
- Side-by-side performance comparison
- Strategy emergence visualization
- Population dynamics analysis
- Learning curve comparisons across agent types

## 🚀 Quick Start

```bash
cd hybrid-curiosity-arena
pip install -r requirements.txt
python main.py
```

Open your browser to `http://localhost:7000` for the unified interface.

## 🧪 Experimental Scenarios

1. **Pure Populations**: All curious vs. all competitive agents
2. **Mixed Populations**: Various ratios of different agent types
3. **Evolutionary Pressure**: Agents switch strategies based on success
4. **Resource Scarcity**: Strategy adaptation under limited resources
5. **Cooperation Emergence**: Cross-strategy cooperation analysis

## 🎮 Web Interface

### **Unified Dashboard**
- Single interface controlling all agent types
- Real-time population composition adjustment
- Live strategy switching for individual agents
- Comparative performance charts

### **Experiment Designer**
- Drag-and-drop agent composition
- Custom reward function mixing
- Real-time parameter adjustment
- Export/import experiment configurations

## 📊 Research Applications

This project addresses fundamental questions in multi-agent RL:

- **Strategy Dominance**: Which learning approach wins in direct competition?
- **Emergent Cooperation**: Do different agent types learn to cooperate?
- **Adaptive Learning**: Can agents optimally switch strategies?
- **Population Dynamics**: How do mixed populations evolve?
- **Resource Efficiency**: Which strategies are most efficient?

## 🏗️ Architecture

```
hybrid-curiosity-arena/
├── agents/           # Different agent implementations
├── environment/      # Unified multi-agent environment
├── training/         # Hybrid training systems
├── web/             # Unified web interface
├── experiments/     # Pre-configured experiments
├── models/          # Saved agent models
├── config/          # Configuration files
└── utils/           # Shared utilities
```

## 🔬 Technical Innovation

- **First-of-its-kind** combination of PPO + ICM with A3C competitive learning
- **Real-time strategy switching** during training
- **Population-level analytics** for multi-paradigm learning
- **Unified observation space** across different learning algorithms
- **Dynamic reward systems** that adapt to agent composition

## 📈 Performance Metrics

- Individual agent performance across strategies
- Population-level efficiency and adaptation
- Strategy emergence and dominance patterns
- Cooperation vs. competition dynamics
- Resource utilization efficiency

## 🤝 Contributing

This project represents cutting-edge research in multi-agent reinforcement learning. Contributions welcome for:

- New agent strategies
- Advanced analytics
- Experiment configurations
- Performance optimizations
- Research applications

## 📚 References

- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning
- Schulman, J., et al. (2017). Proximal policy optimization algorithms
- Pathak, D., et al. (2017). Curiosity-driven exploration by self-supervised prediction
- Tampuu, A., et al. (2017). Multiagent deep reinforcement learning

## 🎯 Future Directions

- Hierarchical multi-agent learning
- Communication protocols between agent types
- Transfer learning across strategies
- Evolutionary algorithm integration
- Real-world application domains

---

**Status**: Active Development | **License**: MIT | **Python**: 3.8+

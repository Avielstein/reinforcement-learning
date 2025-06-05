# Multi-Agent Genetic Team Survival System

A sophisticated reinforcement learning environment where teams of agents evolve through genetic algorithms to survive in a competitive multi-agent world. Watch as neural network policies evolve, teams grow and split, and complex survival strategies emerge naturally through pure evolutionary pressure.

## 🧬 Key Features

### **Genetic Evolution of Neural Networks**
- **Policy Evolution**: Actual neural network weights evolve through genetic operations
- **Team-Based Learning**: Agents share successful policies within teams
- **Dynamic Population**: Teams grow, shrink, split, and get eliminated based on performance
- **Emergent Behaviors**: Complex strategies emerge without explicit programming

### **Multi-Agent Survival Environment**
- **Team Competition**: Multiple teams compete for survival in shared environment
- **Combat System**: Agents can attack enemies within range
- **Spatial Dynamics**: Movement, positioning, and territorial control matter
- **Real-time Simulation**: Fast, efficient simulation with live visualization

### **Advanced Team Dynamics**
- **Dynamic Team Sizes**: Teams adjust size based on survival success
- **Team Splitting**: Successful large teams split into competing sub-teams
- **Policy Sharing**: Teammates share learned strategies
- **Elimination Pressure**: Poor-performing teams face extinction

### **Comprehensive Analytics**
- **Real-time Metrics**: Live tracking of team performance and evolution
- **Behavioral Analysis**: Automatic detection of emergent strategies
- **Population Statistics**: Detailed insights into evolutionary trends
- **Web Interface**: Interactive visualization and control

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch numpy matplotlib flask
```

### Run Simple Demo
```bash
cd survival-genetic-teams
python experiments/simple_demo.py
```

### Expected Output
```
🧬 Multi-Agent Genetic Team Survival System
==================================================
Configuration:
  Initial Teams: 3
  Starting Team Size: 3
  Episode Length: 200
  World Size: 800x600

🚀 Starting simulation...
Running 5 episodes to demonstrate evolution...

🧬 Starting Episode 1
   Teams: 3
   Total Agents: 9

🏆 Episode 1 Results:
   Team 0: 2/3 survived (66.7%)
   Team 1: 1/3 survived (33.3%)
   Team 2: 3/3 survived (100.0%)

🧬 Starting Episode 2
   Teams: 3
   Total Agents: 10
...
```

## 🏗️ System Architecture

### **Modular Design**
```
survival-genetic-teams/
├── core/                    # Core configuration and data types
│   ├── config.py           # Global configuration settings
│   ├── types.py            # Data structures and enums
│   └── metrics.py          # Performance tracking
├── agents/                  # Individual agent components
│   ├── policy_net.py       # Neural network policies
│   ├── survival_agent.py   # Agent with learning capabilities
│   └── experience.py       # Experience storage and replay
├── teams/                   # Team management and evolution
│   ├── team.py             # Team coordination and genetics
│   └── population.py       # Population-level evolution
├── environment/             # Simulation environment
│   └── survival_env.py     # Multi-agent survival world
├── simulation/              # Simulation execution
│   └── episode_runner.py   # Episode management and control
├── visualization/           # Real-time visualization
│   └── web_interface/      # Web-based interface
└── experiments/             # Example scripts and demos
    └── simple_demo.py      # Quick demonstration
```

## 🎯 How It Works

### **1. Team Initialization**
- Multiple teams spawn with random neural network policies
- Each team starts with configurable number of agents
- Teams spawn in different areas of the environment

### **2. Survival Competition**
- Agents observe nearby teammates and enemies
- Neural networks decide movement and attack actions
- Combat occurs when agents are in range
- Survival depends on avoiding damage and eliminating threats

### **3. Genetic Evolution**
- **Within Episodes**: Agents learn and share policies with teammates
- **Between Episodes**: Teams evolve based on survival rates
  - High survival → team growth
  - Low survival → team shrinkage
  - Zero survival → team elimination
  - Large teams → splitting into competing sub-teams

### **4. Emergent Behaviors**
Through pure survival pressure, teams develop:
- **Territorial Control**: Claiming and defending areas
- **Pack Hunting**: Coordinated attacks on isolated enemies
- **Evasion Tactics**: Hit-and-run strategies
- **Formation Fighting**: Protective positioning
- **Resource Competition**: Controlling advantageous positions

## ⚙️ Configuration

### **Team Settings**
```python
config = Config()
config.INITIAL_TEAMS = 5          # Starting number of teams
config.MIN_TEAM_SIZE = 2          # Minimum agents per team
config.MAX_TEAM_SIZE = 12         # Maximum before splitting
config.STARTING_TEAM_SIZE = 4     # Initial team size
```

### **Evolution Parameters**
```python
config.SURVIVAL_THRESHOLD = 0.3   # Min survival rate to avoid shrinking
config.GROWTH_THRESHOLD = 0.7     # Survival rate needed for growth
config.MUTATION_RATE = 0.1        # Probability of weight mutations
config.POLICY_SHARING_STRENGTH = 0.3  # Team policy blending strength
```

### **Environment Settings**
```python
config.WORLD_WIDTH = 800          # Environment width
config.WORLD_HEIGHT = 600         # Environment height
config.EPISODE_LENGTH = 1000      # Maximum steps per episode
config.AGENT_ATTACK_RANGE = 15.0  # Combat range
config.AGENT_VISION_RANGE = 50.0  # Observation range
```

## 📊 Monitoring Evolution

### **Real-time Console Output**
```
🧬 Starting Episode 15
   Teams: 4
   Total Agents: 18

📊 Episode 15 Complete
   Duration: 847 steps
   Team 0: 3/4 survived (75.0%)
   Team 1: 2/5 survived (40.0%)
   Team 2: 4/4 survived (100.0%)
   Team 3: 1/5 survived (20.0%)
   💀 Eliminated: Teams []
   ⚔️  Combat: 156 events, 12 kills
```

### **Performance Metrics**
- **Survival Rates**: Team-by-team survival statistics
- **Population Dynamics**: Team sizes and generation tracking
- **Combat Statistics**: Damage dealt, kills, tactical effectiveness
- **Evolutionary Trends**: Growth, decline, and diversity measures

### **Behavioral Insights**
- **Team Strategies**: Aggressive, defensive, evasive patterns
- **Adaptation Rates**: How quickly teams respond to pressure
- **Diversity Scores**: Behavioral variation within teams
- **Innovation Detection**: Emergence of new strategies

## 🌐 Web Interface (Coming Soon)

### **Real-time Visualization**
- Live simulation display with team colors
- Agent movement trails and combat indicators
- Interactive parameter adjustment
- Performance charts and statistics

### **Control Features**
- Start/pause/reset simulation
- Adjust evolution parameters on-the-fly
- Save/load population states
- Export evolution data

## 🔬 Research Applications

### **Evolutionary Algorithms**
- Study genetic evolution of neural networks
- Compare different selection pressures
- Analyze population dynamics and diversity
- Test novel genetic operators

### **Multi-Agent Systems**
- Emergent coordination without communication
- Competitive vs cooperative evolution
- Scalability of genetic approaches
- Arms races and counter-strategies

### **Behavioral Analysis**
- Automatic strategy classification
- Adaptation to environmental changes
- Innovation and creativity in AI systems
- Social dynamics in artificial populations

## 🎮 Advanced Usage

### **Custom Experiments**
```python
from core.config import Config
from simulation.episode_runner import EpisodeRunner

# Create custom configuration
config = Config()
config.INITIAL_TEAMS = 8
config.EPISODE_LENGTH = 2000

# Run extended evolution
runner = EpisodeRunner(config)
runner.start_simulation(max_episodes=100)

# Save results
runner.save_simulation_state("results/experiment_1")
```

### **Parameter Studies**
```python
# Test different mutation rates
for mutation_rate in [0.05, 0.1, 0.2]:
    config.MUTATION_RATE = mutation_rate
    runner = EpisodeRunner(config)
    results = runner.run_single_episode()
    # Analyze results...
```

### **Behavioral Analysis**
```python
# Get detailed team statistics
performance = runner.get_performance_summary()
evolution_insights = performance['evolution']

print(f"Growing teams: {evolution_insights['growing_teams']}")
print(f"Average diversity: {evolution_insights['average_diversity']}")
```

## 🔧 Customization

### **Adding New Behaviors**
Extend the reward function in `agents/survival_agent.py`:
```python
def calculate_reward(self, observation, action, damage_dealt, damage_taken):
    reward = 0.0
    
    # Add custom reward components
    reward += self.custom_behavior_reward(observation, action)
    
    return reward
```

### **New Evolution Strategies**
Modify genetic operations in `teams/team.py`:
```python
def _apply_genetic_operations(self, survival_rate):
    # Custom evolution logic
    self.custom_evolution_strategy(survival_rate)
```

### **Environment Modifications**
Add environmental features in `environment/survival_env.py`:
```python
def add_environmental_hazards(self):
    # Implement custom environmental challenges
    pass
```

## 📈 Expected Discoveries

Based on the system design, you should observe:

### **Early Evolution (Episodes 1-20)**
- Random movement and combat
- High casualty rates
- Rapid team size fluctuations

### **Intermediate Evolution (Episodes 20-100)**
- Emergence of basic strategies
- Team specialization begins
- Stable territorial behaviors

### **Advanced Evolution (Episodes 100+)**
- Sophisticated coordination
- Counter-strategies and arms races
- Complex emergent behaviors
- Stable population dynamics

## 🤝 Contributing

This system is designed for extensibility:

1. **New Agent Types**: Add different neural architectures
2. **Environment Variants**: Create new survival challenges
3. **Evolution Algorithms**: Implement alternative genetic operators
4. **Visualization Tools**: Enhance real-time monitoring
5. **Analysis Methods**: Develop new behavioral metrics

## 📝 License

MIT License - Use freely for research and educational purposes.

---

**Ready to watch evolution in action? Start with the simple demo and witness the emergence of complex survival strategies through pure genetic evolution!**

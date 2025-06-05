# Genetic Radar Evolution

A genetic algorithm system that evolves neural network policies for tactical radar combat. Instead of evolving simple trait parameters, this system evolves the actual neural network weights and decision-making strategies, allowing for the emergence of complex behaviors.

## 🧬 Key Features

### Neural Network Evolution
- **Policy Networks**: Multi-headed neural networks that make tactical decisions
- **Genetic Operations**: Crossover and mutation directly on network weights
- **Specialized Architectures**: Different starting strategies (aggressive, defensive, scout)
- **Emergent Behaviors**: Complex strategies emerge from simple neural evolution

### Behavior Analysis
- **Real-time Tracking**: Monitor how behaviors evolve over generations
- **Strategy Classification**: Automatic categorization of emergent strategies
- **Diversity Metrics**: Measure behavioral diversity within species
- **Innovation Detection**: Track when new behaviors appear

### Modular Architecture
- **Core**: Configuration, genetics, and base classes
- **Agents**: Neural networks, agents, and behavior analysis
- **Environment**: Combat simulation and physics
- **Evolution**: Genetic algorithms and selection pressure
- **Visualization**: Real-time plots and analysis charts
- **Experiments**: Complete experiment runners

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch numpy matplotlib
```

### Run the Simple Demo
```bash
cd genetic-radar-evolution
python simple_demo.py
```

### Run a Full Evolution Experiment (Advanced)
```bash
cd genetic-radar-evolution/experiments
python run_evolution.py
# Note: Full experiment requires additional setup
```

## 📊 What You'll See

### Evolution in Action
- **Generation Progress**: Watch species fitness evolve over time
- **Emergent Behaviors**: See new strategies appear naturally
- **Species Competition**: Observe which approaches succeed
- **Behavioral Reports**: Detailed analysis every 5 generations

### Example Output
```
🧬 Generation 15
----------------------------------------
⚔️  Battle 1/3
⚔️  Battle 2/3
⚔️  Battle 3/3
   Alpha Hunters: 245.67
   Beta Defenders: 189.23
   Gamma Scouts: 198.45
   🧬 Major evolution for Beta Defenders

📊 Behavior Analysis (Generation 15):
🧬 GENETIC EVOLUTION BEHAVIOR ANALYSIS REPORT
============================================================

📊 SPECIES: ALPHA
------------------------------
🎯 Dominant Strategies:
   • berserker: 66.7%
   • assault: 33.3%

🌈 Behavioral Diversity: 0.234
🚀 Emergent Behaviors:
   • Trigger Happy
   • Erratic Mover
   • Escalating Aggression
```

## 🔬 System Architecture

### Neural Network Policies
Each agent has a neural network with specialized output heads:
- **Movement Head**: Controls x,y movement decisions
- **Combat Head**: Decides when to fire and target selection
- **Tactical Head**: Manages cooperation and exploration

### Genetic Evolution
- **Crossover**: Combine successful neural networks
- **Mutation**: Add noise to network weights
- **Selection**: Keep top performers, evolve struggling species
- **Elitism**: Preserve best strategies while exploring new ones

### Emergent Behaviors
The system automatically detects and categorizes behaviors:
- **Movement Patterns**: Stationary defender, erratic mover, circling pattern
- **Combat Styles**: Trigger happy, sniper, opportunist, alpha hunter
- **Adaptation**: Learning restraint, escalating aggression, stable strategy
- **Social Behaviors**: Team player, lone wolf, cooperative

## 🎯 Experiment Types

### Fast Experiment (20 generations)
- Quick results for testing and demonstration
- 3 battles per generation
- 4 agents per species
- ~10-15 minutes runtime

### Detailed Experiment (50 generations)
- Comprehensive evolution study
- 5 battles per generation  
- 6 agents per species
- ~30-45 minutes runtime

### Custom Configuration
Modify `core/config.py` to adjust:
- Population sizes
- Mutation rates
- Battle parameters
- Neural network architecture
- Behavior tracking settings

## 📈 Analysis and Visualization

### Automatic Outputs
- **Evolution History**: JSON files with generation data
- **Best Networks**: Saved PyTorch models for each species
- **Behavior Reports**: Human-readable analysis
- **Visualization Plots**: Fitness trends and behavior evolution

### Key Metrics
- **Fitness Scores**: Survival, kills, damage, time-based
- **Behavioral Diversity**: How different agents are within species
- **Strategy Stability**: How consistent strategies remain
- **Innovation Rate**: How often new behaviors emerge

## 🧪 Research Applications

### Studying Emergent Behaviors
- How do complex strategies emerge from simple rules?
- What environmental pressures lead to cooperation vs competition?
- How does population diversity affect innovation?

### Algorithm Development
- Compare different genetic operators
- Test various neural network architectures
- Experiment with selection pressures

### Multi-Agent Systems
- Study coordination without explicit communication
- Observe arms races between competing strategies
- Analyze adaptation to changing environments

## 🔧 Customization

### Adding New Behaviors
Extend `agents/behavior_analyzer.py` to detect new patterns:
```python
def _analyze_custom_pattern(self, action_history):
    # Your custom behavior detection logic
    if some_condition:
        return 'new_behavior_name'
    return None
```

### Modifying Evolution
Adjust genetic operators in `evolution/genetic_evolution.py`:
```python
def custom_mutation(self, network, rate, strength):
    # Your custom mutation strategy
    pass
```

### New Species
Add species in `experiments/run_evolution.py`:
```python
{
    'id': 'delta',
    'name': 'Delta Specialists', 
    'strategy': 'sniper',
    'population_size': 6
}
```

## 📝 File Structure

```
genetic-radar-evolution/
├── core/                    # Base classes and configuration
│   ├── config.py           # Experiment configuration
│   ├── genetics.py         # Genetic trait system (legacy)
│   └── species.py          # Species management
├── agents/                  # Neural agents and behavior
│   ├── policy_network.py   # Evolving neural networks
│   ├── neural_agent.py     # Agent with neural decision making
│   └── behavior_analyzer.py # Behavior pattern detection
├── environment/             # Combat simulation
│   └── combat_environment.py # Battle physics and rules
├── evolution/               # Genetic algorithms
│   └── genetic_evolution.py # Evolution management
├── visualization/           # Plotting and analysis
│   └── evolution_visualizer.py # Charts and graphs
├── experiments/             # Experiment runners
│   └── run_evolution.py    # Main experiment script
└── data/                   # Output directory for results
```

## 🎮 Interactive Features

### Real-time Monitoring
- Watch fitness scores evolve
- See behavior reports every 5 generations
- Track emergent strategy development

### Saved Results
- Best neural networks for each species
- Complete evolution history
- Behavioral analysis reports
- Visualization plots

## 🔬 Expected Discoveries

Based on the system design, you might observe:

### Emergent Strategies
- **Pack Hunters**: Coordinated group attacks
- **Ambush Predators**: Patient, precise strikers
- **Swarm Tactics**: Fast, overwhelming numbers
- **Defensive Formations**: Protective positioning

### Evolutionary Arms Races
- Aggressive species driving defensive adaptations
- Scout species forcing counter-reconnaissance
- Accuracy improvements leading to evasion tactics

### Behavioral Innovation
- Novel movement patterns
- Unexpected cooperation strategies
- Creative target selection methods
- Adaptive learning behaviors

## 🚀 Getting Started

1. **Clone and Setup**:
   ```bash
   cd genetic-radar-evolution
   pip install torch numpy matplotlib
   ```

2. **Run First Experiment**:
   ```bash
   cd experiments
   python run_evolution.py
   ```

3. **Choose Fast Experiment** (option 1) for quick results

4. **Watch the Evolution**: Monitor console output for behavior reports

5. **Analyze Results**: Check the `data/` directory for saved outputs

The system will automatically detect and report emergent behaviors as they evolve, giving you insights into how complex strategies can emerge from simple neural network evolution!

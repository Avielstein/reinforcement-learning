# Quick Start Guide

## 🚀 Run the Demo

### Console Demo
```bash
cd survival-genetic-teams
python experiments/simple_demo.py
```

### Web Interface (Recommended!)
```bash
cd survival-genetic-teams
python start_web_interface.py
```
Then open your browser to: http://localhost:5002

## 🧬 What You'll See

The demo runs 5 episodes showing:
- **Team Growth**: Teams that survive well grow in size
- **Genetic Evolution**: Neural network policies evolve through survival pressure
- **Real-time Stats**: Live tracking of team performance and population dynamics

## 📊 Sample Output

```
🧬 Multi-Agent Genetic Team Survival System
==================================================
Configuration:
  Initial Teams: 3
  Starting Team Size: 3
  Episode Length: 200
  World Size: 800x600

🧬 Starting Episode 1
   Teams: 3
   Total Agents: 9

📊 Episode 1 Complete
   Duration: 200 steps
   Team 0: 3/3 survived (100.0%)
   Team 1: 3/3 survived (100.0%)
   Team 2: 3/3 survived (100.0%)
```

## 🎯 Key Features Demonstrated

1. **Dynamic Team Sizes**: Teams grow when they perform well
2. **Genetic Evolution**: Neural networks evolve through survival pressure
3. **Real-time Monitoring**: Live console output showing evolution progress
4. **Performance Tracking**: Detailed statistics on team and agent performance

## 🔧 Customization

Edit `experiments/simple_demo.py` to adjust:
- Number of teams
- Episode length
- Team size limits
- World dimensions

## 📈 Next Steps

- Run longer simulations to see more complex evolution
- Experiment with different parameters
- Add web visualization (framework ready)
- Implement new survival scenarios

The system is fully modular and extensible - perfect for research and experimentation!

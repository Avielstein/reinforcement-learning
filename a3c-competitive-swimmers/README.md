# A3C Competitive Swimmers

A multi-agent reinforcement learning implementation using A3C (Asynchronous Actor-Critic) with trust regions for competitive swimming agents. This project explores parallel learning, shared knowledge transfer, and competitive dynamics in a waterworld environment.

## Overview

This project implements A3C with trust region constraints for multiple swimming agents that compete for food resources while sharing learned experiences. The system combines:

- **A3C (Asynchronous Actor-Critic)** for parallel learning across multiple agents
- **Trust Region Policy Optimization** for stable policy updates
- **Multi-Agent Competition** where agents compete for limited food resources
- **Shared Learning** where agents exchange knowledge at the end of learning cycles
- **Competitive Dynamics** with reward based on food collection efficiency

## Key Features

- **Multiple Swimming Agents**: 2-8 agents competing in the same environment
- **Asynchronous Learning**: Each agent learns independently in parallel
- **Trust Region Constraints**: Stable policy updates preventing catastrophic changes
- **Knowledge Sharing**: Agents share learned policies at cycle boundaries
- **Competitive Rewards**: First-to-food gets the reward, encouraging strategic behavior
- **Real-time Visualization**: Watch agents learn and compete in real-time
- **Performance Analytics**: Track individual and collective learning progress

## Architecture

### Agent System
- **A3C Workers**: Independent learning threads for each agent
- **Shared Global Network**: Common policy network updated asynchronously
- **Trust Region Constraints**: KL divergence limits for policy updates
- **Experience Sharing**: Periodic knowledge transfer between agents

### Environment
- **Multi-Agent Waterworld**: Extended from single-agent implementations
- **Competitive Food Spawning**: Limited food resources create competition
- **Collision Detection**: Agents can interact and influence each other
- **Dynamic Difficulty**: Food spawn rate adapts to agent performance

### Learning Process
1. **Parallel Exploration**: Multiple agents explore simultaneously
2. **Asynchronous Updates**: Each agent updates global network independently
3. **Trust Region Validation**: Policy changes constrained by KL divergence
4. **Periodic Sharing**: Agents exchange learned strategies at intervals
5. **Competitive Adaptation**: Agents adapt to each other's strategies

## Usage

### Quick Start
```bash
cd a3c-competitive-swimmers
pip install -r requirements.txt
python main.py
```

### Headless Training
```bash
# Train with 4 agents for 100k steps
python train_headless.py --agents 4 --steps 100000

# Custom configuration
python train_headless.py --agents 6 --lr 0.0003 --trust-region-coef 0.01
```

### Web Interface
```bash
python main.py
# Open http://localhost:5000
```

### Experiments
```bash
# Compare different agent counts
python experiments/agent_scaling.py

# Trust region ablation study
python experiments/trust_region_analysis.py

# Competition vs cooperation analysis
python experiments/competitive_dynamics.py
```

## Configuration

### Agent Parameters
- `num_agents`: Number of competing agents (2-8)
- `learning_rate`: Learning rate for policy updates
- `trust_region_coef`: KL divergence constraint strength
- `sharing_interval`: Steps between knowledge sharing
- `entropy_coef`: Exploration vs exploitation balance

### Environment Parameters
- `food_spawn_rate`: Rate of food appearance
- `max_food_items`: Maximum concurrent food items
- `agent_collision`: Enable/disable agent-agent interactions
- `competitive_rewards`: Reward structure (winner-take-all vs shared)

### Training Parameters
- `max_steps`: Total training steps
- `update_frequency`: Global network update frequency
- `save_interval`: Model checkpoint frequency
- `eval_episodes`: Episodes for performance evaluation

## Research Questions

This implementation explores several key research areas:

1. **Parallel Learning Efficiency**: How does A3C compare to single-agent learning?
2. **Trust Region Benefits**: Do trust regions improve multi-agent stability?
3. **Knowledge Sharing Impact**: How does shared learning affect individual performance?
4. **Competitive Dynamics**: How do agents adapt to competitive pressure?
5. **Scaling Properties**: How does performance change with agent count?

## Expected Behaviors

### Early Training
- Random exploration with frequent collisions
- Inefficient food collection
- High policy variance between agents

### Mid Training
- Emergence of basic food-seeking strategies
- Reduced random movement
- Beginning of competitive behaviors

### Late Training
- Sophisticated competitive strategies
- Efficient food collection paths
- Adaptive responses to other agents
- Stable policy convergence

## Technical Implementation

### A3C Architecture
- **Actor Network**: Policy function π(a|s)
- **Critic Network**: Value function V(s)
- **Shared Parameters**: Global network updated asynchronously
- **Local Workers**: Independent agent threads

### Trust Region Constraints
- **KL Divergence Monitoring**: Track policy change magnitude
- **Adaptive Step Sizes**: Adjust learning rate based on KL divergence
- **Rollback Mechanism**: Revert updates that violate trust region

### Multi-Agent Environment
- **Shared State Space**: All agents observe same environment
- **Individual Actions**: Each agent controls independently
- **Competitive Rewards**: Zero-sum food collection rewards
- **Communication Channel**: Optional agent-to-agent communication

## Performance Metrics

- **Individual Agent Performance**: Food collection rate per agent
- **Collective Efficiency**: Total food collected by all agents
- **Learning Stability**: Policy variance over time
- **Competitive Balance**: Distribution of rewards across agents
- **Convergence Speed**: Time to stable performance

## References

- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.
- Schulman, J., et al. (2015). Trust region policy optimization. ICML.
- Tampuu, A., et al. (2017). Multiagent deep reinforcement learning with extremely sparse rewards.
- Foerster, J., et al. (2018). Counterfactual multi-agent policy gradients. AAAI.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Flask (for web interface)
- Threading support for parallel agents

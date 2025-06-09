# 🐠 TD Fish Follow: Temporal Difference Learning for Continuous Control

A comprehensive implementation of Temporal Difference (TD) learning methods applied to a continuous control task where an artificial fish learns to follow moving targets in a simulated aquatic environment.

## 📋 Table of Contents

1. [Technical Introduction](#technical-introduction)
2. [Core ML/RL Concepts & Literature](#core-mlrl-concepts--literature)
3. [Training Instructions](#training-instructions)
4. [Web Interface Usage](#web-interface-usage)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)

---

## 🧠 Technical Introduction

### What We're Doing

This project implements and compares multiple **Temporal Difference (TD) learning** algorithms for continuous control in a fish-following task. Unlike traditional reinforcement learning approaches that rely on Monte Carlo methods or policy gradients, TD learning provides a middle ground that can learn from incomplete episodes while maintaining sample efficiency.

### Why This Matters

**Continuous Control Challenges:**
- **High-dimensional action spaces**: Unlike discrete actions, continuous control requires learning smooth policies over infinite action spaces
- **Temporal credit assignment**: Determining which actions led to rewards across time
- **Sample efficiency**: Learning effective policies with minimal environment interaction
- **Exploration vs exploitation**: Balancing discovery of new behaviors with exploitation of known good actions

**Real-World Applications:**
- **Autonomous underwater vehicles (AUVs)**: Target tracking and navigation
- **Robotics**: Smooth manipulation and locomotion
- **Biological modeling**: Understanding animal behavior and learning
- **Game AI**: Realistic character movement and behavior

### Technical Innovation

Our implementation advances the state-of-the-art by:

1. **Multi-method TD comparison**: Direct comparison of TD(0), TD(λ), and n-step TD in the same environment
2. **Continuous action spaces**: Extension of traditional discrete TD methods to continuous control
3. **Experience replay integration**: Combining TD learning with modern deep RL techniques
4. **Rich observation spaces**: 15-dimensional observations including temporal features
5. **Multiple target patterns**: Testing generalization across different movement behaviors

---

## 📚 Core ML/RL Concepts & Literature

### Temporal Difference Learning

**Core Concept**: TD learning estimates value functions by bootstrapping from current estimates, updating predictions based on observed rewards and subsequent state values.

**Key Papers:**
- **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement learning: An introduction* (2nd ed.). MIT Press.
  - The foundational textbook covering all TD methods
- **Sutton, R. S. (1988)**. Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1), 9-44.
  - Original TD learning paper

### TD(λ) and Eligibility Traces

**Core Concept**: Eligibility traces provide a mechanism for temporal credit assignment, allowing updates to affect multiple previous states with exponentially decaying influence.

**Key Papers:**
- **Singh, S. P., & Sutton, R. S. (1996)**. Reinforcement learning with replacing eligibility traces. *Machine Learning*, 22(1-3), 123-158.
- **Sutton, R. S., & Barto, A. G. (1998)**. Reinforcement Learning: An Introduction. Chapter 7: Eligibility Traces.

### Experience Replay

**Core Concept**: Store and replay past experiences to improve sample efficiency and break temporal correlations in training data.

**Key Papers:**
- **Lin, L. J. (1992)**. Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning*, 8(3-4), 293-321.
  - Original experience replay concept
- **Mnih, V., et al. (2015)**. Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
  - DQN paper that popularized experience replay in deep RL

### Prioritized Experience Replay

**Core Concept**: Sample experiences for replay based on their TD error magnitude, focusing learning on the most surprising transitions.

**Key Papers:**
- **Schaul, T., et al. (2016)**. Prioritized experience replay. *ICLR 2016*.

### Actor-Critic Methods

**Core Concept**: Combine value function estimation (critic) with direct policy optimization (actor) for continuous control.

**Key Papers:**
- **Konda, V. R., & Tsitsiklis, J. N. (2000)**. Actor-critic algorithms. *NIPS 2000*.
- **Sutton, R. S., et al. (2000)**. Policy gradient methods for reinforcement learning with function approximation. *NIPS 2000*.

### Continuous Control

**Core Concept**: Extend RL methods from discrete to continuous action spaces using policy gradients and deterministic policies.

**Key Papers:**
- **Lillicrap, T. P., et al. (2016)**. Continuous control with deep reinforcement learning. *ICLR 2016*.
  - DDPG: Deterministic Policy Gradient for continuous control
- **Schulman, J., et al. (2017)**. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
  - PPO: Stable policy gradient method
- **Haarnoja, T., et al. (2018)**. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML 2018*.

### Target Networks

**Core Concept**: Use separate target networks to stabilize learning by providing consistent targets during training.

**Key Papers:**
- **Mnih, V., et al. (2015)**. Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- **Van Hasselt, H., Guez, A., & Silver, D. (2016)**. Deep reinforcement learning with double Q-learning. *AAAI 2016*.

---

## 🚀 Training Instructions

### Basic Training

1. **Train from scratch with default settings:**
```bash
cd td-fish-follow
python -m training.train_agent
```

2. **Train with specific TD method:**
```bash
# TD(0) - One-step temporal difference
python -m training.train_agent --method td_0

# TD(λ) - Eligibility traces (default λ=0.9)
python -m training.train_agent --method td_lambda --lambda 0.9

# N-step TD - Multi-step returns
python -m training.train_agent --method n_step_td --n_steps 5
```

3. **Train with different target patterns:**
```bash
# Random walk (default)
python -m training.train_agent --pattern random_walk

# Circular movement
python -m training.train_agent --pattern circular

# Figure-8 pattern
python -m training.train_agent --pattern figure8

# Zigzag movement
python -m training.train_agent --pattern zigzag

# Spiral pattern
python -m training.train_agent --pattern spiral
```

### Advanced Training Options

```bash
# Custom training configuration
python -m training.train_agent \
    --method td_lambda \
    --pattern circular \
    --episodes 1000 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --replay_buffer_size 100000 \
    --target_update_freq 100 \
    --save_interval 50
```

### Model Persistence

**Saving Models:**
```python
# During training, models are automatically saved to:
# models/checkpoints/agent_episode_X.pt
# models/best_agent.pt (best performing model)

# Manual saving:
from core.agent import TDFishAgent
agent.save('models/my_trained_agent.pt')
```

**Loading Models:**
```python
from core.agent import TDFishAgent
from config.environment import EnvironmentConfig
from config.td_config import TDConfig

# Load configuration
env_config = EnvironmentConfig()
td_config = TDConfig()

# Create and load agent
agent = TDFishAgent(env_config, td_config)
agent.load('models/my_trained_agent.pt')

# Use for evaluation
agent.set_training_mode(False)
```

**Evaluation:**
```bash
# Evaluate trained model
python -m training.evaluate_agent --model_path models/best_agent.pt --episodes 100
```

### Comparing TD Methods

Run comprehensive comparison across all TD methods:

```bash
# Compare all methods on all patterns
python -m experiments.td_comparison --episodes 500

# Results saved to: results/td_comparison_results.pt
# Plots saved to: results/td_comparison_*.png
```

---

## 🌐 Web Interface Usage

### Starting the Interface

1. **Launch the web server:**
```bash
cd td-fish-follow
python start_web_interface.py
```

2. **Open your browser to:** `http://localhost:5000`

### Understanding the Interface

**Main Components:**

1. **Fish Tank Visualization**
   - 🐠 **Red fish**: The learning agent
   - 🟡 **Yellow target**: The goal to follow
   - **Trail**: Shows fish movement history
   - **Real-time updates**: 20 FPS visualization

2. **Control Panel**
   - **Target Pattern**: Choose movement pattern for target
   - **TD Method**: Select learning algorithm
   - **Training Controls**: Start/stop/reset buttons
   - **Manual Step**: Step through learning one action at a time

3. **Statistics Display**
   - **Episode**: Current episode number
   - **Step**: Current step within episode
   - **Distance**: Current distance to target
   - **Reward**: Cumulative episode reward
   - **TD Error**: Magnitude of temporal difference error
   - **Action**: Current action values [x, y]

4. **Training Progress**
   - **Average Reward**: Rolling average over last 10 episodes
   - **Average Distance**: Rolling average distance to target
   - **Best Distance**: Best average distance achieved
   - **Learning Rate**: Current learning rate

5. **Live Charts**
   - **Episode Rewards**: Training progress over time
   - **Distance to Target**: Following performance over time

### Interpreting the Learning Process

**What to Watch For:**

1. **Initial Random Behavior**: Fish moves randomly, high TD errors
2. **Exploration Phase**: Fish begins to move toward target occasionally
3. **Learning Phase**: TD errors decrease, distance to target improves
4. **Convergence**: Smooth following behavior, low TD errors

**Performance Indicators:**

- **Decreasing TD Error**: Model predictions becoming more accurate
- **Decreasing Distance**: Fish getting better at following target
- **Increasing Reward**: Overall performance improvement
- **Smoother Movement**: Less erratic, more purposeful actions

### Comparing Methods

Use the interface to compare different TD methods:

1. **Start with TD(0)**: Observe baseline performance
2. **Switch to TD(λ)**: Notice faster learning due to eligibility traces
3. **Try N-Step TD**: Compare multi-step vs single-step learning
4. **Test Different Patterns**: See how methods generalize

---

## 📁 Project Structure

```
td-fish-follow/
├── config/                 # Configuration files
│   ├── environment.py      # Environment parameters
│   ├── td_config.py        # TD learning configuration
│   └── training.py         # Training parameters
├── core/                   # Core implementation
│   ├── environment.py      # Fish tank environment
│   ├── agent.py           # Main TD learning agent
│   ├── td_learner.py      # TD learning algorithms
│   └── replay_buffer.py   # Experience replay
├── models/                 # Neural network architectures
│   ├── networks.py        # Basic network components
│   ├── td_critic.py       # TD-specific critic networks
│   └── policy_net.py      # Policy networks
├── training/               # Training scripts
│   ├── train_agent.py     # Main training script
│   └── evaluate_agent.py  # Evaluation script
├── experiments/            # Research experiments
│   └── td_comparison.py   # Compare TD methods
├── web/                    # Web interface
│   ├── index.html         # Main interface
│   └── static/
│       ├── styles.css     # Styling
│       └── script.js      # Interactive functionality
├── visualization/          # Visualization tools
│   └── web_interface.py   # Web server backend
└── results/               # Training results and plots
```

---

## 🛠 Installation & Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Flask (for web interface)
- Matplotlib (for plotting)

### Installation

1. **Clone or navigate to the project:**
```bash
cd td-fish-follow
```

2. **Install dependencies:**
```bash
pip install torch numpy flask matplotlib
```

3. **Verify installation:**
```bash
python test_basic.py
```

### Quick Start

1. **Test the system:**
```bash
python test_basic.py
```

2. **Start training:**
```bash
python -m training.train_agent --episodes 100
```

3. **Launch web interface:**
```bash
python start_web_interface.py
```

4. **Open browser to:** `http://localhost:5000`

---

## 🎯 Research Applications

This implementation is designed for research and education in:

- **Temporal Difference Learning**: Compare different TD methods
- **Continuous Control**: Study RL in continuous action spaces
- **Sample Efficiency**: Analyze learning speed across methods
- **Transfer Learning**: Test generalization across target patterns
- **Biological Modeling**: Model animal learning and behavior

## 📊 Expected Results

Based on RL theory and empirical studies:

- **TD(λ)**: Fastest learning due to eligibility traces
- **N-Step TD**: Good balance of bias and variance
- **TD(0)**: Most stable but potentially slower learning
- **Experience Replay**: Improved sample efficiency across all methods

## 🤝 Contributing

This project is designed for research and educational use. Feel free to:

- Experiment with different network architectures
- Add new target movement patterns
- Implement additional TD variants
- Extend to multi-agent scenarios

---

## 📄 License

This project is for educational and research purposes. Please cite relevant papers when using this code in academic work.

## 🙏 Acknowledgments

Built upon foundational work in reinforcement learning by Sutton, Barto, and the broader RL community. Special thanks to the authors of the key papers referenced above.

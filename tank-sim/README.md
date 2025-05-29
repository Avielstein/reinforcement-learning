# Tank Simulation Projects

This directory contains multiple reinforcement learning projects focused on tank/aquarium environments with different objectives and learning scenarios.

## Project Structure

### `utils/`
Shared utility modules used across all tank simulation projects:
- `constants.py` - Environment and training parameters
- `environment.py` - Base fish tank environment and water current classes
- `models.py` - Neural network architectures (Actor-Critic)
- `trainer.py` - A2C training algorithm implementation
- `visualization.py` - Training visualization and testing tools

### `fish-to-center/`
**Objective**: Train a fish agent to navigate to and stay at the center of the tank while dealing with dynamic water currents.

**Features**:
- A2C (Advantage Actor-Critic) implementation
- Physics-based fish movement with realistic water dynamics
- Dynamic water currents that evolve over time
- Real-time training visualization
- Interactive web-based simulator
- Multiple trained models available

**Files**:
- `train-RL-fish.ipynb` - Main training notebook
- `fish-tank-rl.html` - Interactive web simulator
- `*.pt` - Saved model files
- Various experiment notebooks

### `dot-follow/`
**Objective**: [To be implemented] Train a fish agent to follow a moving dot/target around the tank.

**Planned Features**:
- Moving target that the fish must track
- Different target movement patterns (circular, random, figure-8, etc.)
- Reward based on proximity to moving target
- Potential for curriculum learning with increasing target speeds

## Getting Started

### Prerequisites
```bash
pip install torch gymnasium matplotlib numpy jupyter
```

### Running Projects

#### Fish-to-Center
```bash
cd fish-to-center
jupyter notebook train-RL-fish.ipynb
# Or open fish-tank-rl.html in browser for web demo
```

#### Dot-Follow
```bash
cd dot-follow
# [Implementation coming soon]
```

## Shared Utilities

All projects use the shared `utils/` package. To use in your notebooks:

```python
import sys
sys.path.append('../')  # Add parent directory to path
from utils import FishTankEnv, A2CLearner, run_training_visualization
```

## Contributing

When adding new tank simulation projects:
1. Create a new subdirectory under `tank-sim/`
2. Use the shared `utils/` package when possible
3. Add project-specific documentation
4. Update this README with the new project description

# Squid Robot Simulation

A clean, simple implementation of a squid-like robot using jet propulsion for underwater navigation.

## Project Structure

```
gymnasium-exploration/
├── README.md           # This file
├── requirements.txt    # Minimal dependencies
└── squid_robot.py     # Main squid robot implementation
```

## Features

- **Jet Propulsion**: Water intake and expulsion system
- **Steerable Nozzle**: ±45° steering range from straight back
- **Water Management**: Tank capacity with refill/consumption mechanics
- **Top-down View**: Clean 2D navigation focused on motion mechanics
- **Gymnasium Interface**: Standard RL environment for future training

## Quick Start

1. **Setup Environment**
   ```bash
   cd gymnasium-exploration
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Simulation**
   ```bash
   python squid_robot.py
   ```

3. **Controls**
   - `W/S`: Thrust power (0-1)
   - `A/D`: Nozzle steering (-45° to +45°)
   - `ESC`: Quit

## Design Philosophy

This implementation focuses on:
- **Simplicity**: Clean, readable code without unnecessary complexity
- **Core Mechanics**: Essential squid motion physics only
- **Extensibility**: Easy to add features incrementally
- **No Bloat**: Minimal dependencies and file structure

## Next Steps

- Add obstacles and navigation challenges
- Implement reinforcement learning training
- Add sensors and perception systems
- Create multi-agent scenarios

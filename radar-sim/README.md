# Tactical Radar Simulator

A React-based tactical radar simulation environment for testing multi-agent reinforcement learning algorithms in military-style scenarios. This project explores strategic decision-making and situational awareness in complex, multi-entity environments.

## Overview

This simulator provides a tactical radar interface for studying multi-agent RL in strategic scenarios. It complements the other projects in the collection by focusing on discrete strategic decisions rather than continuous control, and serves as a testbed for hierarchical RL and multi-agent coordination algorithms that could be integrated with the survival-genetic-teams project.

## Usage

```bash
cd radar-sim
npm install
npm start
# Open http://localhost:3000
```

### Development
```bash
npm run build    # Production build
npm run test     # Run tests
```

## Requirements

- Node.js 16+
- React 18+
- TypeScript
- Modern web browser with Canvas support

## References

- Tampuu, A., et al. (2017). Multiagent deep reinforcement learning with extremely sparse rewards. arXiv preprint.
- Foerster, J., et al. (2018). Counterfactual multi-agent policy gradients. AAAI.

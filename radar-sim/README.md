# Tactical Radar Combat Simulator

A real-time tactical combat simulation that visualizes radar-based detection and combat between multiple teams of units. Watch as units autonomously patrol, detect enemies, engage in combat, and employ evasive maneuvers.


## Features

- Dynamic radar detection visualization
- Multiple autonomous unit behaviors (patrolling, pursuing, firing, evading)
- Three-team combat simulation
- Projectile physics with leading target calculations
- Visual effects for explosions and projectile trails
- Adjustable simulation parameters
- Battle log to track combat events

## Prerequisites

- Node.js (v14 or later)
- npm or yarn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tactical-radar-simulator.git
   cd tactical-radar-simulator
   ```

2. Install the dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm start
   # or
   yarn start
   ```

4. Open your browser and navigate to `http://localhost:3000`

## Usage

### Controls

- **Pause/Resume**: Toggle simulation running state
- **Reset**: Restart the simulation with current team configurations
- **Simulation Speed**: Adjust the speed of the simulation (0.5x to 3x)
- **Team Sliders**: Set the number of units for each team (0-5)

### Team Colors

- **Red Team**: Spawns in the upper left
- **Blue Team**: Spawns in the lower right
- **Green Team**: Spawns in the lower left

### Unit Behaviors

Units automatically transition between different states:

- **Patrolling**: Random movement around the map
- **Pursuing**: Detected an enemy and moving toward them
- **Firing**: Within range of an enemy and shooting
- **Evading**: Low health and trying to escape

## Technical Details

This simulator is built with:
- React for UI rendering
- Canvas API for graphics
- Real-time simulation using requestAnimationFrame
- Tailwind CSS for styling

## Customization

You can modify the `tactical-radar-simulator.tsx` file to adjust simulation parameters:

- Unit speed, health, and radar range
- Map size and dimensions
- Combat mechanics like reload time and damage
- Starting positions for each team

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

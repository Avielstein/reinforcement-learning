import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional


class Agent:
    """Individual swimming agent in the competitive environment."""
    
    def __init__(self, agent_id: int, x: float, y: float, world_width: float, world_height: float):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 8.0
        self.max_speed = 15.0  # Much faster agents
        self.world_width = world_width
        self.world_height = world_height
        
        # Performance tracking
        self.food_collected = 0
        self.total_reward = 0.0
        self.steps_alive = 0
        
        # Sensor configuration (30 rays like PPO curious fish)
        self.num_rays = 30
        self.ray_length = 100.0
        
    def update_position(self, action: int, dt: float = 1.0):
        """Update agent position based on action."""
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        force_magnitude = 8.0  # Much stronger forces
        
        if action == 0:  # Up
            self.vy -= force_magnitude
        elif action == 1:  # Down
            self.vy += force_magnitude
        elif action == 2:  # Left
            self.vx -= force_magnitude
        elif action == 3:  # Right
            self.vx += force_magnitude
        
        # Apply friction
        friction = 0.95
        self.vx *= friction
        self.vy *= friction
        
        # Limit speed
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > self.max_speed:
            self.vx = (self.vx / speed) * self.max_speed
            self.vy = (self.vy / speed) * self.max_speed
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Handle wall collisions with repulsion
        wall_repulsion = 10.0
        if self.x < self.radius:
            self.x = self.radius
            self.vx = abs(self.vx) * 0.8
        elif self.x > self.world_width - self.radius:
            self.x = self.world_width - self.radius
            self.vx = -abs(self.vx) * 0.8
            
        if self.y < self.radius:
            self.y = self.radius
            self.vy = abs(self.vy) * 0.8
        elif self.y > self.world_height - self.radius:
            self.y = self.world_height - self.radius
            self.vy = -abs(self.vy) * 0.8
        
        self.steps_alive += 1
    
    def get_sensors(self, food_items: List, other_agents: List, poison_items: List = None) -> np.ndarray:
        """Get 30-ray sensor readings like PPO curious fish."""
        sensors = []
        
        for i in range(self.num_rays):
            angle = (2 * math.pi * i) / self.num_rays
            
            # Ray direction
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
            
            # Initialize sensor values
            food_dist = 1.0  # Normalized distance (1.0 = max range)
            food_detected = 0.0
            poison_dist = 1.0
            poison_detected = 0.0
            agent_dist = 1.0
            agent_detected = 0.0
            
            # Check intersections along the ray
            for step in range(1, int(self.ray_length)):
                ray_x = self.x + ray_dx * step
                ray_y = self.y + ray_dy * step
                
                # Check bounds
                if (ray_x < 0 or ray_x > self.world_width or 
                    ray_y < 0 or ray_y > self.world_height):
                    break
                
                # Check food items
                for food in food_items:
                    dist_to_food = math.sqrt((ray_x - food.x)**2 + (ray_y - food.y)**2)
                    if dist_to_food < food.radius and food_detected == 0.0:
                        food_dist = step / self.ray_length
                        food_detected = 1.0
                
                # Check other agents
                for other_agent in other_agents:
                    if other_agent.agent_id != self.agent_id:
                        dist_to_agent = math.sqrt((ray_x - other_agent.x)**2 + (ray_y - other_agent.y)**2)
                        if dist_to_agent < other_agent.radius and agent_detected == 0.0:
                            agent_dist = step / self.ray_length
                            agent_detected = 1.0
                
                # Check poison items if they exist
                if poison_items:
                    for poison in poison_items:
                        dist_to_poison = math.sqrt((ray_x - poison.x)**2 + (ray_y - poison.y)**2)
                        if dist_to_poison < poison.radius and poison_detected == 0.0:
                            poison_dist = step / self.ray_length
                            poison_detected = 1.0
            
            # Add sensor readings for this ray
            sensors.extend([
                1.0 - food_dist,      # Closer food = higher value
                food_detected,         # Food present on this ray
                1.0 - agent_dist,      # Closer agent = higher value
                agent_detected,        # Agent present on this ray
                1.0 - poison_dist if poison_items else 0.0  # Poison distance (if enabled)
            ])
        
        # Add proprioception (agent's own velocity)
        sensors.extend([
            self.vx / self.max_speed,  # Normalized velocity x
            self.vy / self.max_speed   # Normalized velocity y
        ])
        
        return np.array(sensors, dtype=np.float32)


class FoodItem:
    """Food item that agents compete for."""
    
    def __init__(self, x: float, y: float, value: float = 1.0):
        self.x = x
        self.y = y
        self.radius = 6.0
        self.value = value
        self.collected = False


class BouncingItem:
    """Base class for bouncing items (food, poison, obstacles)."""
    
    def __init__(self, x: float, y: float, radius: float, world_width: float, world_height: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.world_width = world_width
        self.world_height = world_height
        
        # Random velocity - much faster
        self.vx = random.uniform(-8.0, 8.0)
        self.vy = random.uniform(-8.0, 8.0)
        
    def update(self, dt: float = 1.0):
        """Update position and handle wall bouncing."""
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Bounce off walls
        if self.x <= self.radius or self.x >= self.world_width - self.radius:
            self.vx = -self.vx
            self.x = max(self.radius, min(self.world_width - self.radius, self.x))
            
        if self.y <= self.radius or self.y >= self.world_height - self.radius:
            self.vy = -self.vy
            self.y = max(self.radius, min(self.world_height - self.radius, self.y))


class BouncingFood(BouncingItem):
    """Bouncing food item (red apple)."""
    
    def __init__(self, x: float, y: float, world_width: float, world_height: float):
        super().__init__(x, y, 6.0, world_width, world_height)
        self.value = 1.0
        self.collected = False


class BouncingPoison(BouncingItem):
    """Bouncing poison item (green poison)."""
    
    def __init__(self, x: float, y: float, world_width: float, world_height: float):
        super().__init__(x, y, 6.0, world_width, world_height)
        self.value = -1.0


class BouncingObstacle(BouncingItem):
    """Bouncing obstacle (neutral, just blocks movement)."""
    
    def __init__(self, x: float, y: float, world_width: float, world_height: float):
        super().__init__(x, y, 8.0, world_width, world_height)
        # Obstacles move slower
        self.vx *= 0.5
        self.vy *= 0.5


class CompetitiveWaterworld:
    """
    Multi-agent competitive waterworld environment.
    
    Multiple agents compete for limited food resources.
    Includes bouncing food, poison, and obstacles like Karpathy's demo.
    """
    
    def __init__(self, 
                 num_agents: int = 4,
                 world_width: float = 400.0,
                 world_height: float = 400.0,
                 max_food_items: int = 20,
                 max_poison_items: int = 20,
                 max_obstacles: int = 0,  # No gray obstacles
                 food_spawn_rate: float = 0.03,
                 poison_spawn_rate: float = 0.03,
                 competitive_rewards: bool = True):
        
        self.num_agents = num_agents
        self.world_width = world_width
        self.world_height = world_height
        self.max_food_items = max_food_items
        self.max_poison_items = max_poison_items
        self.max_obstacles = max_obstacles
        self.food_spawn_rate = food_spawn_rate
        self.poison_spawn_rate = poison_spawn_rate
        self.competitive_rewards = competitive_rewards
        
        # Initialize agents
        self.agents = []
        for i in range(num_agents):
            # Spawn agents in different corners to avoid initial collisions
            if i == 0:
                x, y = 50, 50
            elif i == 1:
                x, y = world_width - 50, 50
            elif i == 2:
                x, y = 50, world_height - 50
            elif i == 3:
                x, y = world_width - 50, world_height - 50
            else:
                # Random positions for additional agents
                x = random.uniform(50, world_width - 50)
                y = random.uniform(50, world_height - 50)
            
            agent = Agent(i, x, y, world_width, world_height)
            self.agents.append(agent)
        
        # Initialize bouncing items
        self.food_items = []
        self.poison_items = []
        self.obstacles = []
        
        # Environment state
        self.step_count = 0
        self.total_food_spawned = 0
        self.total_food_collected = 0
        
        # Performance tracking
        self.agent_rewards = [0.0] * num_agents
        self.agent_food_counts = [0] * num_agents
        
        # State space calculation (same as PPO curious fish)
        # 30 rays * 5 values per ray + 2 proprioception = 152
        self.state_dim = 30 * 5 + 2
        self.action_dim = 4  # Up, Down, Left, Right
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial states for all agents."""
        # Reset agents to starting positions
        for i, agent in enumerate(self.agents):
            if i == 0:
                agent.x, agent.y = 50, 50
            elif i == 1:
                agent.x, agent.y = self.world_width - 50, 50
            elif i == 2:
                agent.x, agent.y = 50, self.world_height - 50
            elif i == 3:
                agent.x, agent.y = self.world_width - 50, self.world_height - 50
            else:
                agent.x = random.uniform(50, self.world_width - 50)
                agent.y = random.uniform(50, self.world_height - 50)
            
            agent.vx = agent.vy = 0.0
            agent.food_collected = 0
            agent.total_reward = 0.0
            agent.steps_alive = 0
        
        # Clear all items
        self.food_items.clear()
        self.poison_items.clear()
        self.obstacles.clear()
        
        # Reset counters
        self.step_count = 0
        self.total_food_spawned = 0
        self.total_food_collected = 0
        self.agent_rewards = [0.0] * self.num_agents
        self.agent_food_counts = [0] * self.num_agents
        
        # Spawn initial items
        self._spawn_initial_items()
        
        # Return initial states
        return self._get_all_states()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """
        Execute one step for all agents.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            states: New states for all agents
            rewards: Rewards for all agents
            dones: Done flags for all agents
            info: Additional information
        """
        self.step_count += 1
        
        # Update agent positions
        for i, agent in enumerate(self.agents):
            agent.update_position(actions[i])
        
        # Update bouncing items
        for food in self.food_items:
            food.update()
        for poison in self.poison_items:
            poison.update()
        for obstacle in self.obstacles:
            obstacle.update()
        
        # Check food collection
        rewards = self._check_food_collection()
        
        # Check poison collision
        poison_penalties = self._check_poison_collision()
        for i in range(self.num_agents):
            rewards[i] += poison_penalties[i]
        
        # Spawn new items
        self._maybe_spawn_food()
        self._maybe_spawn_poison()
        
        # Update performance tracking
        for i in range(self.num_agents):
            self.agent_rewards[i] += rewards[i]
        
        # Get new states
        states = self._get_all_states()
        
        # Check if episode is done (for now, never done - continuous environment)
        dones = [False] * self.num_agents
        
        # Create info dictionary
        info = {
            'step_count': self.step_count,
            'total_food_spawned': self.total_food_spawned,
            'total_food_collected': self.total_food_collected,
            'agent_food_counts': self.agent_food_counts.copy(),
            'agent_rewards': self.agent_rewards.copy(),
            'num_food_items': len(self.food_items),
            'num_poison_items': len(self.poison_items),
            'num_obstacles': len(self.obstacles),
            'num_agents': self.num_agents
        }
        
        return states, rewards, dones, info
    
    def _get_all_states(self) -> List[np.ndarray]:
        """Get sensor states for all agents."""
        states = []
        for agent in self.agents:
            state = agent.get_sensors(self.food_items, self.agents, self.poison_items)
            states.append(state)
        return states
    
    def _check_food_collection(self) -> List[float]:
        """Check if any agents collected food and assign rewards."""
        rewards = [0.0] * self.num_agents
        food_to_remove = []
        
        for food_idx, food in enumerate(self.food_items):
            if food.collected:
                continue
                
            # Check which agents are close enough to collect this food
            collecting_agents = []
            for agent in self.agents:
                dist = math.sqrt((agent.x - food.x)**2 + (agent.y - food.y)**2)
                if dist < (agent.radius + food.radius):
                    collecting_agents.append(agent)
            
            if collecting_agents:
                if self.competitive_rewards:
                    # Only the first agent gets the reward (competitive)
                    winner = collecting_agents[0]  # Could be random or based on distance
                    rewards[winner.agent_id] = food.value
                    winner.food_collected += 1
                    self.agent_food_counts[winner.agent_id] += 1
                else:
                    # All agents get partial reward (cooperative)
                    reward_per_agent = food.value / len(collecting_agents)
                    for agent in collecting_agents:
                        rewards[agent.agent_id] = reward_per_agent
                        agent.food_collected += 1
                        self.agent_food_counts[agent.agent_id] += 1
                
                food.collected = True
                food_to_remove.append(food_idx)
                self.total_food_collected += 1
        
        # Remove collected food
        for idx in reversed(food_to_remove):
            del self.food_items[idx]
        
        return rewards
    
    def _check_poison_collision(self) -> List[float]:
        """Check poison collisions and assign penalties."""
        penalties = [0.0] * self.num_agents
        
        if not self.poison_items:
            return penalties
        
        poison_to_remove = []
        
        for poison_idx, poison in enumerate(self.poison_items):
            for agent in self.agents:
                dist = math.sqrt((agent.x - poison.x)**2 + (agent.y - poison.y)**2)
                if dist < (agent.radius + poison.radius):
                    penalties[agent.agent_id] = -1.0  # Poison penalty
                    poison_to_remove.append(poison_idx)
                    break
        
        # Remove consumed poison
        for idx in reversed(poison_to_remove):
            del self.poison_items[idx]
        
        return penalties
    
    def _spawn_initial_items(self):
        """Spawn initial bouncing items."""
        # Spawn bouncing food
        for _ in range(self.max_food_items // 2):
            self._spawn_bouncing_food()
        
        # Spawn bouncing poison
        for _ in range(self.max_poison_items // 2):
            self._spawn_bouncing_poison()
        
        # Spawn obstacles
        for _ in range(self.max_obstacles):
            self._spawn_obstacle()
    
    def _maybe_spawn_food(self):
        """Maybe spawn new bouncing food based on spawn rate."""
        if len(self.food_items) < self.max_food_items and random.random() < self.food_spawn_rate:
            self._spawn_bouncing_food()
    
    def _spawn_bouncing_food(self):
        """Spawn a bouncing food item at random location."""
        x = random.uniform(30, self.world_width - 30)
        y = random.uniform(30, self.world_height - 30)
        
        food = BouncingFood(x, y, self.world_width, self.world_height)
        self.food_items.append(food)
        self.total_food_spawned += 1
    
    def _maybe_spawn_poison(self):
        """Maybe spawn new bouncing poison based on spawn rate."""
        if len(self.poison_items) < self.max_poison_items and random.random() < self.poison_spawn_rate:
            self._spawn_bouncing_poison()
    
    def _spawn_bouncing_poison(self):
        """Spawn a bouncing poison item."""
        x = random.uniform(30, self.world_width - 30)
        y = random.uniform(30, self.world_height - 30)
        
        poison = BouncingPoison(x, y, self.world_width, self.world_height)
        self.poison_items.append(poison)
    
    def _spawn_obstacle(self):
        """Spawn a bouncing obstacle."""
        x = random.uniform(40, self.world_width - 40)
        y = random.uniform(40, self.world_height - 40)
        
        obstacle = BouncingObstacle(x, y, self.world_width, self.world_height)
        self.obstacles.append(obstacle)
    
    def get_state_action_dims(self) -> Tuple[int, int]:
        """Get state and action space dimensions."""
        return self.state_dim, self.action_dim
    
    def render_state(self) -> Dict:
        """Return current state for visualization."""
        return {
            'agents': [
                {
                    'id': agent.agent_id,
                    'x': agent.x,
                    'y': agent.y,
                    'vx': agent.vx,
                    'vy': agent.vy,
                    'radius': agent.radius,
                    'food_collected': agent.food_collected,
                    'total_reward': agent.total_reward
                }
                for agent in self.agents
            ],
            'food_items': [
                {
                    'x': food.x,
                    'y': food.y,
                    'vx': food.vx,
                    'vy': food.vy,
                    'radius': food.radius,
                    'value': food.value
                }
                for food in self.food_items
            ],
            'poison_items': [
                {
                    'x': poison.x,
                    'y': poison.y,
                    'vx': poison.vx,
                    'vy': poison.vy,
                    'radius': poison.radius
                }
                for poison in self.poison_items
            ],
            'obstacles': [
                {
                    'x': obstacle.x,
                    'y': obstacle.y,
                    'vx': obstacle.vx,
                    'vy': obstacle.vy,
                    'radius': obstacle.radius
                }
                for obstacle in self.obstacles
            ],
            'world_width': self.world_width,
            'world_height': self.world_height,
            'step_count': self.step_count
        }


if __name__ == "__main__":
    # Test the environment
    env = CompetitiveWaterworld(num_agents=4)
    
    print("Testing Competitive Waterworld Environment")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    
    # Reset and test
    states = env.reset()
    print(f"Initial states shape: {[state.shape for state in states]}")
    
    # Test a few steps
    for step in range(5):
        actions = [random.randint(0, 3) for _ in range(env.num_agents)]
        states, rewards, dones, info = env.step(actions)
        
        print(f"Step {step + 1}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {[f'{r:.2f}' for r in rewards]}")
        print(f"  Food items: {info['num_food_items']}")
        print(f"  Total food collected: {info['total_food_collected']}")
    
    print("Environment test completed successfully!")

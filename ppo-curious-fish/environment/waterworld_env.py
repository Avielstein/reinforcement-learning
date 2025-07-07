import numpy as np
import random
from .entities import Fish, Food, Poison

class FishWaterworld:
    """
    Fish Waterworld environment matching Karpathy's demo structure.
    
    - 152-dimensional state space (30 sensors × 5 values + 2 proprioception)
    - 4-dimensional continuous action space (left, right, up, down thrust)
    - Reward: +1 for food, -1 for poison
    """
    
    def __init__(self, width=400, height=300, num_food=25, num_poison=25):
        self.width = width
        self.height = height
        self.num_food = num_food
        self.num_poison = num_poison
        
        # Create fish agent
        self.fish = Fish(width//2, height//2, width, height)
        
        # Create food and poison entities
        self.food_items = []
        self.poison_items = []
        
        # Environment state
        self.step_count = 0
        self.episode_reward = 0.0
        self.dt = 1.0 / 60.0  # 60 FPS like Karpathy's demo
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset fish to center
        self.fish.position = np.array([self.width//2, self.height//2], dtype=np.float32)
        self.fish.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.fish.angle = 0.0
        
        # Clear and recreate entities
        self.food_items.clear()
        self.poison_items.clear()
        
        # Create food items
        for _ in range(self.num_food):
            x = random.uniform(20, self.width - 20)
            y = random.uniform(20, self.height - 20)
            # Ensure food doesn't spawn too close to fish
            while np.linalg.norm([x - self.fish.position[0], y - self.fish.position[1]]) < 50:
                x = random.uniform(20, self.width - 20)
                y = random.uniform(20, self.height - 20)
            self.food_items.append(Food(x, y, self.width, self.height))
        
        # Create poison items
        for _ in range(self.num_poison):
            x = random.uniform(20, self.width - 20)
            y = random.uniform(20, self.height - 20)
            # Ensure poison doesn't spawn too close to fish
            while np.linalg.norm([x - self.fish.position[0], y - self.fish.position[1]]) < 50:
                x = random.uniform(20, self.width - 20)
                y = random.uniform(20, self.height - 20)
            self.poison_items.append(Poison(x, y, self.width, self.height))
        
        # Reset counters
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self.get_state()
    
    def step(self, action):
        """
        Take one step in the environment.
        
        Args:
            action: 4-dimensional array [left, right, up, down] thrust values
            
        Returns:
            state: 152-dimensional observation
            reward: scalar reward
            done: boolean indicating episode termination
            info: dictionary with additional information
        """
        # Apply action to fish
        self.fish.apply_action(action)
        
        # Update all entities
        all_entities = [self.fish] + self.food_items + self.poison_items
        for entity in all_entities:
            entity.update(self.dt)
        
        # Check collisions and calculate reward
        reward = 0.0
        food_eaten = 0
        poison_eaten = 0
        
        # Check food collisions
        for i, food in enumerate(self.food_items[:]):  # Copy list to avoid modification during iteration
            if self.fish.collides_with(food):
                reward += 1.0
                food_eaten += 1
                # Respawn food at random location
                x = random.uniform(20, self.width - 20)
                y = random.uniform(20, self.height - 20)
                self.food_items[i] = Food(x, y, self.width, self.height)
        
        # Check poison collisions
        for i, poison in enumerate(self.poison_items[:]):
            if self.fish.collides_with(poison):
                reward -= 1.0
                poison_eaten += 1
                # Respawn poison at random location
                x = random.uniform(20, self.width - 20)
                y = random.uniform(20, self.height - 20)
                self.poison_items[i] = Poison(x, y, self.width, self.height)
        
        # Update counters
        self.step_count += 1
        self.episode_reward += reward
        
        # Get new state
        state = self.get_state()
        
        # Episode never ends in this environment (like Karpathy's demo)
        done = False
        
        # Additional info
        info = {
            'food_eaten': food_eaten,
            'poison_eaten': poison_eaten,
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'fish_position': self.fish.position.copy(),
            'fish_velocity': self.fish.velocity.copy(),
            'fish_angle': self.fish.angle
        }
        
        return state, reward, done, info
    
    def get_state(self):
        """Get the current 152-dimensional state observation."""
        all_entities = self.food_items + self.poison_items
        return self.fish.get_sensor_data(all_entities)
    
    def get_visualization_data(self):
        """Get data for web visualization."""
        return {
            'fish': {
                'x': float(self.fish.position[0]),
                'y': float(self.fish.position[1]),
                'angle': float(self.fish.angle),
                'radius': float(self.fish.radius),
                'velocity': {
                    'x': float(self.fish.velocity[0]),
                    'y': float(self.fish.velocity[1])
                }
            },
            'food': [
                {
                    'x': float(food.position[0]),
                    'y': float(food.position[1]),
                    'radius': float(food.radius),
                    'velocity': {
                        'x': float(food.velocity[0]),
                        'y': float(food.velocity[1])
                    }
                }
                for food in self.food_items
            ],
            'poison': [
                {
                    'x': float(poison.position[0]),
                    'y': float(poison.position[1]),
                    'radius': float(poison.radius),
                    'velocity': {
                        'x': float(poison.velocity[0]),
                        'y': float(poison.velocity[1])
                    }
                }
                for poison in self.poison_items
            ],
            'sensors': self._get_sensor_visualization()
        }
    
    def _get_sensor_visualization(self):
        """Get sensor ray data for visualization."""
        sensor_rays = []
        all_entities = self.food_items + self.poison_items
        
        for i, sensor_angle in enumerate(self.fish.sensor_angles):
            world_angle = self.fish.angle + sensor_angle
            ray_dir = np.array([np.cos(world_angle), np.sin(world_angle)])
            
            # Find intersection point
            closest_dist = self.fish.sensor_range
            hit_entity = None
            
            for entity in all_entities:
                to_entity = entity.position - self.fish.position
                projection = np.dot(to_entity, ray_dir)
                
                if projection <= 0:
                    continue
                    
                ray_point = self.fish.position + ray_dir * projection
                dist_to_ray = np.linalg.norm(entity.position - ray_point)
                
                if dist_to_ray <= entity.radius and projection < closest_dist:
                    closest_dist = projection
                    hit_entity = entity
            
            # Ray end point
            end_point = self.fish.position + ray_dir * closest_dist
            
            sensor_rays.append({
                'start': {
                    'x': float(self.fish.position[0]),
                    'y': float(self.fish.position[1])
                },
                'end': {
                    'x': float(end_point[0]),
                    'y': float(end_point[1])
                },
                'hit': hit_entity is not None,
                'hit_type': 'food' if isinstance(hit_entity, Food) else 'poison' if isinstance(hit_entity, Poison) else None
            })
        
        return sensor_rays
    
    @property
    def state_dim(self):
        """Dimension of state space."""
        return 152  # 30 sensors × 5 values + 2 proprioception
    
    @property
    def action_dim(self):
        """Dimension of action space."""
        return 4  # left, right, up, down thrust
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

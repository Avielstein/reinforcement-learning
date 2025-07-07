import numpy as np
import math

class Entity:
    """Base class for all entities in the waterworld."""
    
    def __init__(self, x, y, radius=5.0):
        self.position = np.array([x, y], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.radius = radius
        self.alive = True
        
    def update(self, dt):
        """Update entity position based on velocity."""
        self.position += self.velocity * dt
        
    def distance_to(self, other):
        """Calculate distance to another entity."""
        return np.linalg.norm(self.position - other.position)
        
    def collides_with(self, other):
        """Check collision with another entity."""
        return self.distance_to(other) < (self.radius + other.radius)

class Fish(Entity):
    """The fish agent that learns to navigate the waterworld."""
    
    def __init__(self, x, y, world_width=800, world_height=600):
        super().__init__(x, y, radius=8.0)
        self.world_width = world_width
        self.world_height = world_height
        self.angle = 0.0  # Fish orientation
        self.max_speed = 150.0
        self.thrust_power = 200.0
        self.drag = 0.95  # Velocity decay factor
        
        # Sensor system (30 rays like Karpathy's demo)
        self.num_sensors = 30
        self.sensor_range = 120.0
        self.sensor_angles = np.linspace(0, 2*np.pi, self.num_sensors, endpoint=False)
        
    def apply_action(self, action):
        """Apply action to fish (4 actions: left, right, up, down thrust)."""
        # Action is a 4-dimensional vector [left, right, up, down]
        # Each component is between -1 and 1
        thrust = np.array([
            action[1] - action[0],  # right - left = net horizontal thrust
            action[3] - action[2]   # down - up = net vertical thrust
        ]) * self.thrust_power
        
        # Apply thrust to velocity
        self.velocity += thrust * 0.016  # dt = 1/60 seconds
        
        # Limit maximum speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
            
        # Update angle based on velocity direction
        if speed > 1.0:
            self.angle = math.atan2(self.velocity[1], self.velocity[0])
    
    def update(self, dt):
        """Update fish position and handle world boundaries."""
        # Apply drag
        self.velocity *= self.drag
        
        # Update position
        super().update(dt)
        
        # Handle world boundaries (bounce off walls)
        if self.position[0] < self.radius:
            self.position[0] = self.radius
            self.velocity[0] = abs(self.velocity[0])
        elif self.position[0] > self.world_width - self.radius:
            self.position[0] = self.world_width - self.radius
            self.velocity[0] = -abs(self.velocity[0])
            
        if self.position[1] < self.radius:
            self.position[1] = self.radius
            self.velocity[1] = abs(self.velocity[1])
        elif self.position[1] > self.world_height - self.radius:
            self.position[1] = self.world_height - self.radius
            self.velocity[1] = -abs(self.velocity[1])
    
    def get_sensor_data(self, entities):
        """Get 152-dimensional sensor data like Karpathy's demo."""
        sensor_data = []
        
        # 30 sensors, each with 5 values
        for i, sensor_angle in enumerate(self.sensor_angles):
            # World angle (fish angle + sensor angle)
            world_angle = self.angle + sensor_angle
            ray_dir = np.array([np.cos(world_angle), np.sin(world_angle)])
            
            # Cast ray and find closest entity
            closest_dist = self.sensor_range
            closest_type = 0  # 0 = nothing, 1 = food, 2 = poison
            closest_vel = np.array([0.0, 0.0])
            
            for entity in entities:
                if entity == self:
                    continue
                    
                # Vector from fish to entity
                to_entity = entity.position - self.position
                
                # Project onto ray direction
                projection = np.dot(to_entity, ray_dir)
                if projection <= 0:
                    continue  # Entity is behind the ray
                    
                # Distance from ray to entity center
                ray_point = self.position + ray_dir * projection
                dist_to_ray = np.linalg.norm(entity.position - ray_point)
                
                # Check if ray intersects entity
                if dist_to_ray <= entity.radius and projection < closest_dist:
                    closest_dist = projection
                    if isinstance(entity, Food):
                        closest_type = 1
                    elif isinstance(entity, Poison):
                        closest_type = 2
                    closest_vel = entity.velocity.copy()
            
            # Normalize distance
            normalized_dist = closest_dist / self.sensor_range
            
            # 5 values per sensor (matching Karpathy's format)
            sensor_data.extend([
                normalized_dist,
                1.0 if closest_type == 1 else 0.0,  # is food
                1.0 if closest_type == 2 else 0.0,  # is poison
                closest_vel[0] / 100.0,  # normalized velocity x
                closest_vel[1] / 100.0   # normalized velocity y
            ])
        
        # Add proprioception (fish's own velocity) - 2 more dimensions
        sensor_data.extend([
            self.velocity[0] / self.max_speed,  # normalized velocity x
            self.velocity[1] / self.max_speed   # normalized velocity y
        ])
        
        return np.array(sensor_data, dtype=np.float32)  # 152 dimensions total

class Food(Entity):
    """Red food particles that give positive reward."""
    
    def __init__(self, x, y, world_width=800, world_height=600):
        super().__init__(x, y, radius=8.0)
        self.world_width = world_width
        self.world_height = world_height
        self.max_speed = 50.0
        
        # Random initial velocity
        angle = np.random.uniform(0, 2*np.pi)
        speed = np.random.uniform(10, self.max_speed)
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        
    def update(self, dt):
        """Update food position and bounce off walls."""
        super().update(dt)
        
        # Bounce off walls
        if self.position[0] < self.radius or self.position[0] > self.world_width - self.radius:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], self.radius, self.world_width - self.radius)
            
        if self.position[1] < self.radius or self.position[1] > self.world_height - self.radius:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], self.radius, self.world_height - self.radius)

class Poison(Entity):
    """Green poison particles that give negative reward."""
    
    def __init__(self, x, y, world_width=800, world_height=600):
        super().__init__(x, y, radius=8.0)
        self.world_width = world_width
        self.world_height = world_height
        self.max_speed = 30.0
        
        # Random initial velocity (slower than food)
        angle = np.random.uniform(0, 2*np.pi)
        speed = np.random.uniform(5, self.max_speed)
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        
    def update(self, dt):
        """Update poison position and bounce off walls."""
        super().update(dt)
        
        # Bounce off walls
        if self.position[0] < self.radius or self.position[0] > self.world_width - self.radius:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], self.radius, self.world_width - self.radius)
            
        if self.position[1] < self.radius or self.position[1] > self.world_height - self.radius:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], self.radius, self.world_height - self.radius)

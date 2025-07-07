import numpy as np
import math

class Physics:
    """
    Physics simulation for the waterworld environment.
    
    Handles movement dynamics, collisions, and environmental forces.
    """
    
    def __init__(self, world_width=800, world_height=600):
        self.world_width = world_width
        self.world_height = world_height
        self.gravity = 0.0  # No gravity in water
        self.water_resistance = 0.95  # Drag coefficient
        
    def update_entity_physics(self, entity, dt=1.0/60.0):
        """
        Update entity physics (position, velocity, boundaries).
        
        Args:
            entity: Entity to update
            dt: Time step
        """
        # Apply water resistance (drag)
        entity.velocity *= self.water_resistance
        
        # Update position based on velocity
        entity.position += entity.velocity * dt
        
        # Handle boundary collisions
        self.handle_boundary_collision(entity)
    
    def handle_boundary_collision(self, entity):
        """Handle collision with world boundaries."""
        # Left boundary
        if entity.position[0] < entity.radius:
            entity.position[0] = entity.radius
            entity.velocity[0] = abs(entity.velocity[0])  # Bounce
        
        # Right boundary
        elif entity.position[0] > self.world_width - entity.radius:
            entity.position[0] = self.world_width - entity.radius
            entity.velocity[0] = -abs(entity.velocity[0])  # Bounce
        
        # Top boundary
        if entity.position[1] < entity.radius:
            entity.position[1] = entity.radius
            entity.velocity[1] = abs(entity.velocity[1])  # Bounce
        
        # Bottom boundary
        elif entity.position[1] > self.world_height - entity.radius:
            entity.position[1] = self.world_height - entity.radius
            entity.velocity[1] = -abs(entity.velocity[1])  # Bounce
    
    def check_collision(self, entity1, entity2):
        """
        Check if two entities are colliding.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            bool: True if colliding
        """
        distance = np.linalg.norm(entity1.position - entity2.position)
        return distance < (entity1.radius + entity2.radius)
    
    def apply_thrust(self, entity, thrust_vector, max_speed=150.0):
        """
        Apply thrust force to an entity.
        
        Args:
            entity: Entity to apply thrust to
            thrust_vector: [x, y] thrust force
            max_speed: Maximum speed limit
        """
        # Add thrust to velocity
        entity.velocity += thrust_vector
        
        # Limit maximum speed
        speed = np.linalg.norm(entity.velocity)
        if speed > max_speed:
            entity.velocity = entity.velocity / speed * max_speed
    
    def get_distance(self, pos1, pos2):
        """Calculate distance between two positions."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def get_angle_between(self, pos1, pos2):
        """Calculate angle from pos1 to pos2."""
        diff = np.array(pos2) - np.array(pos1)
        return math.atan2(diff[1], diff[0])
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def add_water_current(self, entity, current_strength=0.0, current_direction=0.0):
        """
        Add water current effect to entity.
        
        Args:
            entity: Entity to affect
            current_strength: Strength of current
            current_direction: Direction of current (radians)
        """
        if current_strength > 0:
            current_force = np.array([
                math.cos(current_direction) * current_strength,
                math.sin(current_direction) * current_strength
            ])
            entity.velocity += current_force
    
    def simulate_turbulence(self, entity, turbulence_strength=0.0):
        """
        Add random turbulence to entity movement.
        
        Args:
            entity: Entity to affect
            turbulence_strength: Strength of turbulence
        """
        if turbulence_strength > 0:
            turbulence = np.random.normal(0, turbulence_strength, 2)
            entity.velocity += turbulence

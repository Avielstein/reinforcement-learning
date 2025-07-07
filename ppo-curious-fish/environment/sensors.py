import numpy as np
import math

class FishSensors:
    """
    Fish sensor system for the waterworld environment.
    
    Implements 30-ray vision system matching Karpathy's demo structure.
    Each ray detects: distance, object type, and object velocity.
    """
    
    def __init__(self, num_rays=30, max_range=120.0):
        self.num_rays = num_rays
        self.max_range = max_range
        self.ray_angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        
    def cast_rays(self, fish_position, fish_angle, entities):
        """
        Cast sensor rays and detect entities.
        
        Args:
            fish_position: Fish position [x, y]
            fish_angle: Fish orientation angle
            entities: List of entities to detect
            
        Returns:
            sensor_data: 150-dimensional sensor data (30 rays × 5 values each)
        """
        sensor_data = []
        
        for ray_angle in self.ray_angles:
            # World angle (fish angle + sensor angle)
            world_angle = fish_angle + ray_angle
            ray_direction = np.array([np.cos(world_angle), np.sin(world_angle)])
            
            # Find closest intersection
            closest_distance = self.max_range
            closest_entity = None
            
            for entity in entities:
                # Skip the fish itself
                if hasattr(entity, 'position') and np.array_equal(entity.position, fish_position):
                    continue
                
                # Vector from fish to entity
                to_entity = entity.position - fish_position
                
                # Project onto ray direction
                projection = np.dot(to_entity, ray_direction)
                
                # Skip entities behind the ray
                if projection <= 0:
                    continue
                
                # Distance from ray to entity center
                ray_point = fish_position + ray_direction * projection
                distance_to_ray = np.linalg.norm(entity.position - ray_point)
                
                # Check if ray intersects entity
                if distance_to_ray <= entity.radius and projection < closest_distance:
                    closest_distance = projection
                    closest_entity = entity
            
            # Normalize distance
            normalized_distance = closest_distance / self.max_range
            
            # Determine entity type and velocity
            if closest_entity is None:
                # No entity detected
                entity_type_food = 0.0
                entity_type_poison = 0.0
                entity_velocity = np.array([0.0, 0.0])
            else:
                # Determine entity type
                from .entities import Food, Poison
                entity_type_food = 1.0 if isinstance(closest_entity, Food) else 0.0
                entity_type_poison = 1.0 if isinstance(closest_entity, Poison) else 0.0
                entity_velocity = closest_entity.velocity
            
            # 5 values per ray (matching Karpathy's format)
            sensor_data.extend([
                normalized_distance,           # Distance to closest object
                entity_type_food,             # Is food (red)
                entity_type_poison,           # Is poison (green)
                entity_velocity[0] / 100.0,   # Normalized velocity x
                entity_velocity[1] / 100.0    # Normalized velocity y
            ])
        
        return np.array(sensor_data, dtype=np.float32)
    
    def get_sensor_visualization(self, fish_position, fish_angle, entities):
        """
        Get sensor ray data for visualization.
        
        Returns:
            List of ray data for drawing
        """
        rays = []
        
        for ray_angle in self.ray_angles:
            world_angle = fish_angle + ray_angle
            ray_direction = np.array([np.cos(world_angle), np.sin(world_angle)])
            
            # Find intersection
            closest_distance = self.max_range
            hit_entity = None
            
            for entity in entities:
                if hasattr(entity, 'position') and np.array_equal(entity.position, fish_position):
                    continue
                
                to_entity = entity.position - fish_position
                projection = np.dot(to_entity, ray_direction)
                
                if projection <= 0:
                    continue
                
                ray_point = fish_position + ray_direction * projection
                distance_to_ray = np.linalg.norm(entity.position - ray_point)
                
                if distance_to_ray <= entity.radius and projection < closest_distance:
                    closest_distance = projection
                    hit_entity = entity
            
            # Ray end point
            end_point = fish_position + ray_direction * closest_distance
            
            rays.append({
                'start': {'x': float(fish_position[0]), 'y': float(fish_position[1])},
                'end': {'x': float(end_point[0]), 'y': float(end_point[1])},
                'hit': hit_entity is not None,
                'hit_type': self._get_entity_type(hit_entity)
            })
        
        return rays
    
    def _get_entity_type(self, entity):
        """Get entity type string for visualization."""
        if entity is None:
            return None
        
        from .entities import Food, Poison
        if isinstance(entity, Food):
            return 'food'
        elif isinstance(entity, Poison):
            return 'poison'
        else:
            return 'unknown'

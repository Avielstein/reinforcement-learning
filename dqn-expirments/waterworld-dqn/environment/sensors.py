"""Sensor system for agent perception in WaterWorld."""

import math
from typing import List, Tuple, Optional
from .entities import Agent, Item

class SensorReading:
    """Single sensor reading result."""
    
    def __init__(self, distance: float, item_type: str, hit_position: Optional[Tuple[float, float]] = None):
        self.distance = distance  # Normalized 0-1
        self.item_type = item_type  # 'good', 'bad', 'wall', 'none'
        self.hit_position = hit_position  # For visualization


class SensorSystem:
    """Raycast sensor system for agent perception."""
    
    def __init__(self, sensor_count: int, sensor_range: float, angle_span: float):
        self.sensor_count = sensor_count
        self.sensor_range = sensor_range
        self.angle_span = angle_span
        
        # Pre-calculate sensor angles
        self.sensor_angles = []
        if sensor_count == 1:
            self.sensor_angles = [0.0]
        else:
            for i in range(sensor_count):
                angle = (i / (sensor_count - 1)) * angle_span - (angle_span / 2)
                self.sensor_angles.append(angle)
    
    def sense(self, agent: Agent, good_items: List[Item], bad_items: List[Item], 
              world_width: float, world_height: float) -> List[SensorReading]:
        """Perform sensor sweep and return readings."""
        readings = []
        
        for sensor_angle in self.sensor_angles:
            # Calculate absolute sensor direction
            absolute_angle = agent.angle + sensor_angle
            
            # Cast ray and find nearest hit
            reading = self._cast_ray(agent, absolute_angle, good_items, bad_items, 
                                   world_width, world_height)
            readings.append(reading)
        
        return readings
    
    def _cast_ray(self, agent: Agent, angle: float, good_items: List[Item], 
                  bad_items: List[Item], world_width: float, world_height: float) -> SensorReading:
        """Cast a single ray and return the nearest hit."""
        ray_dx = math.cos(angle)
        ray_dy = math.sin(angle)
        
        nearest_distance = self.sensor_range
        nearest_type = 'none'
        hit_pos = None
        
        # Check wall intersections
        wall_distance = self._ray_wall_intersection(agent.x, agent.y, ray_dx, ray_dy, 
                                                   world_width, world_height)
        if wall_distance < nearest_distance:
            nearest_distance = wall_distance
            nearest_type = 'wall'
            hit_pos = (agent.x + ray_dx * wall_distance, agent.y + ray_dy * wall_distance)
        
        # Check good items
        for item in good_items:
            if item.active:
                distance = self._ray_circle_intersection(agent.x, agent.y, ray_dx, ray_dy,
                                                       item.x, item.y, item.radius)
                if distance is not None and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_type = 'good'
                    hit_pos = (agent.x + ray_dx * distance, agent.y + ray_dy * distance)
        
        # Check bad items
        for item in bad_items:
            if item.active:
                distance = self._ray_circle_intersection(agent.x, agent.y, ray_dx, ray_dy,
                                                       item.x, item.y, item.radius)
                if distance is not None and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_type = 'bad'
                    hit_pos = (agent.x + ray_dx * distance, agent.y + ray_dy * distance)
        
        # Normalize distance to 0-1 range
        normalized_distance = nearest_distance / self.sensor_range
        
        return SensorReading(normalized_distance, nearest_type, hit_pos)
    
    def _ray_wall_intersection(self, start_x: float, start_y: float, 
                              ray_dx: float, ray_dy: float,
                              world_width: float, world_height: float) -> float:
        """Calculate distance to nearest wall intersection."""
        distances = []
        
        # Left wall (x = 0)
        if ray_dx < 0:
            t = -start_x / ray_dx
            if t > 0:
                y = start_y + ray_dy * t
                if 0 <= y <= world_height:
                    distances.append(t)
        
        # Right wall (x = world_width)
        if ray_dx > 0:
            t = (world_width - start_x) / ray_dx
            if t > 0:
                y = start_y + ray_dy * t
                if 0 <= y <= world_height:
                    distances.append(t)
        
        # Bottom wall (y = 0)
        if ray_dy < 0:
            t = -start_y / ray_dy
            if t > 0:
                x = start_x + ray_dx * t
                if 0 <= x <= world_width:
                    distances.append(t)
        
        # Top wall (y = world_height)
        if ray_dy > 0:
            t = (world_height - start_y) / ray_dy
            if t > 0:
                x = start_x + ray_dx * t
                if 0 <= x <= world_width:
                    distances.append(t)
        
        return min(distances) if distances else self.sensor_range
    
    def _ray_circle_intersection(self, start_x: float, start_y: float,
                                ray_dx: float, ray_dy: float,
                                circle_x: float, circle_y: float, 
                                circle_radius: float) -> Optional[float]:
        """Calculate ray-circle intersection distance."""
        # Vector from ray start to circle center
        to_circle_x = circle_x - start_x
        to_circle_y = circle_y - start_y
        
        # Project circle center onto ray
        projection = to_circle_x * ray_dx + to_circle_y * ray_dy
        
        # If projection is negative, ray points away from circle
        if projection < 0:
            return None
        
        # Find closest point on ray to circle center
        closest_x = start_x + ray_dx * projection
        closest_y = start_y + ray_dy * projection
        
        # Distance from circle center to closest point on ray
        distance_to_ray = math.sqrt((circle_x - closest_x)**2 + (circle_y - closest_y)**2)
        
        # Check if ray intersects circle
        if distance_to_ray > circle_radius:
            return None
        
        # Calculate intersection distance
        chord_half_length = math.sqrt(circle_radius**2 - distance_to_ray**2)
        intersection_distance = projection - chord_half_length
        
        return max(0, intersection_distance)

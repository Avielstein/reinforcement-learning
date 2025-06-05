"""
Genetic traits and inheritance system
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class GeneticTraits:
    """Represents the genetic traits of a unit or species"""
    
    # Combat traits
    aggression: float = 0.5      # Tendency to engage in combat
    caution: float = 0.5         # Tendency to avoid danger
    accuracy: float = 0.5        # Weapon accuracy modifier
    reload_speed: float = 0.5    # Reload time modifier
    
    # Movement traits
    speed_bonus: float = 0.5     # Movement speed modifier
    agility: float = 0.5         # Turning and evasion ability
    endurance: float = 0.5       # Energy efficiency
    
    # Tactical traits
    cooperation: float = 0.5     # Tendency to work with allies
    exploration: float = 0.5     # Tendency to explore vs stay put
    radar_efficiency: float = 0.5 # Radar range and accuracy
    target_priority: float = 0.5  # Preference for weak vs strong targets
    
    # Behavioral traits
    risk_tolerance: float = 0.5   # Willingness to take risks
    adaptability: float = 0.5     # Ability to change tactics
    persistence: float = 0.5      # Tendency to pursue targets
    
    # Meta-traits (affect learning and evolution)
    learning_rate: float = 0.5    # How quickly unit adapts
    mutation_resistance: float = 0.5  # Resistance to genetic changes
    
    def __post_init__(self):
        """Ensure all traits are within valid range [0, 1]"""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            setattr(self, field_name, np.clip(value, 0.0, 1.0))
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> 'GeneticTraits':
        """Create a mutated copy of these traits"""
        new_traits = GeneticTraits()
        
        for field_name in self.__dataclass_fields__:
            current_value = getattr(self, field_name)
            
            if random.random() < mutation_rate:
                # Apply mutation
                mutation = random.gauss(0, mutation_strength)
                new_value = np.clip(current_value + mutation, 0.0, 1.0)
                setattr(new_traits, field_name, new_value)
            else:
                setattr(new_traits, field_name, current_value)
        
        return new_traits
    
    def crossover(self, other: 'GeneticTraits', crossover_rate: float = 0.5) -> 'GeneticTraits':
        """Create offspring traits by crossing over with another set of traits"""
        new_traits = GeneticTraits()
        
        for field_name in self.__dataclass_fields__:
            if random.random() < crossover_rate:
                # Take trait from other parent
                value = getattr(other, field_name)
            else:
                # Take trait from this parent
                value = getattr(self, field_name)
            
            setattr(new_traits, field_name, value)
        
        return new_traits
    
    def distance_to(self, other: 'GeneticTraits') -> float:
        """Calculate genetic distance to another set of traits"""
        total_distance = 0.0
        trait_count = 0
        
        for field_name in self.__dataclass_fields__:
            self_value = getattr(self, field_name)
            other_value = getattr(other, field_name)
            total_distance += abs(self_value - other_value)
            trait_count += 1
        
        return total_distance / trait_count if trait_count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert traits to dictionary"""
        return {field_name: getattr(self, field_name) 
                for field_name in self.__dataclass_fields__}
    
    @classmethod
    def from_dict(cls, trait_dict: Dict[str, float]) -> 'GeneticTraits':
        """Create traits from dictionary"""
        return cls(**trait_dict)
    
    @classmethod
    def random(cls) -> 'GeneticTraits':
        """Generate random traits"""
        traits = cls()
        for field_name in traits.__dataclass_fields__:
            setattr(traits, field_name, random.random())
        return traits
    
    @classmethod
    def create_archetype(cls, archetype: str) -> 'GeneticTraits':
        """Create predefined trait archetypes"""
        if archetype == 'aggressive':
            return cls(
                aggression=0.9, caution=0.1, accuracy=0.7, reload_speed=0.8,
                speed_bonus=0.6, agility=0.7, cooperation=0.3, exploration=0.4,
                risk_tolerance=0.9, persistence=0.8
            )
        elif archetype == 'defensive':
            return cls(
                aggression=0.2, caution=0.9, accuracy=0.8, reload_speed=0.4,
                speed_bonus=0.3, agility=0.5, cooperation=0.8, exploration=0.2,
                radar_efficiency=0.9, risk_tolerance=0.2, persistence=0.4
            )
        elif archetype == 'scout':
            return cls(
                aggression=0.4, caution=0.6, accuracy=0.6, reload_speed=0.6,
                speed_bonus=0.9, agility=0.9, cooperation=0.5, exploration=0.9,
                radar_efficiency=0.8, risk_tolerance=0.6, adaptability=0.8
            )
        elif archetype == 'sniper':
            return cls(
                aggression=0.6, caution=0.7, accuracy=0.95, reload_speed=0.3,
                speed_bonus=0.2, agility=0.3, cooperation=0.4, exploration=0.3,
                target_priority=0.8, persistence=0.9, risk_tolerance=0.4
            )
        elif archetype == 'berserker':
            return cls(
                aggression=0.95, caution=0.05, accuracy=0.5, reload_speed=0.9,
                speed_bonus=0.8, agility=0.6, cooperation=0.1, exploration=0.7,
                risk_tolerance=0.95, persistence=0.95, endurance=0.8
            )
        else:  # balanced
            return cls()
    
    def get_behavioral_summary(self) -> str:
        """Get a human-readable summary of behavioral tendencies"""
        behaviors = []
        
        if self.aggression > 0.7:
            behaviors.append("highly aggressive")
        elif self.aggression < 0.3:
            behaviors.append("passive")
        
        if self.caution > 0.7:
            behaviors.append("very cautious")
        elif self.caution < 0.3:
            behaviors.append("reckless")
        
        if self.cooperation > 0.7:
            behaviors.append("team-oriented")
        elif self.cooperation < 0.3:
            behaviors.append("lone wolf")
        
        if self.exploration > 0.7:
            behaviors.append("exploratory")
        elif self.exploration < 0.3:
            behaviors.append("territorial")
        
        if self.accuracy > 0.8:
            behaviors.append("precise shooter")
        elif self.accuracy < 0.4:
            behaviors.append("spray-and-pray")
        
        if self.speed_bonus > 0.7:
            behaviors.append("fast-moving")
        elif self.speed_bonus < 0.3:
            behaviors.append("slow but steady")
        
        return ", ".join(behaviors) if behaviors else "balanced"

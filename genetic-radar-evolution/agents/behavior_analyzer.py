"""
Behavior analysis for emergent strategies in evolved agents
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import matplotlib.pyplot as plt

class BehaviorAnalyzer:
    """Analyzes and categorizes emergent behaviors in evolved agents"""
    
    def __init__(self):
        self.behavior_patterns = defaultdict(list)
        self.strategy_classifications = {}
        self.emergent_behaviors = set()
        
    def analyze_agent_behavior(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual agent behavior patterns"""
        analysis = {
            'primary_strategy': self._classify_strategy(agent_data),
            'behavioral_traits': self._extract_traits(agent_data),
            'emergent_patterns': self._detect_emergent_patterns(agent_data),
            'adaptation_score': self._calculate_adaptation_score(agent_data),
            'uniqueness_score': self._calculate_uniqueness_score(agent_data)
        }
        
        return analysis
    
    def _classify_strategy(self, agent_data: Dict[str, Any]) -> str:
        """Classify agent's primary strategy"""
        metrics = agent_data.get('behavioral_metrics', {})
        
        if not metrics:
            return 'unknown'
        
        aggression = metrics.get('avg_aggression', 0.5)
        exploration = metrics.get('avg_exploration', 0.5)
        cooperation = metrics.get('avg_cooperation', 0.5)
        accuracy = metrics.get('accuracy_ratio', 0.5)
        movement_var = metrics.get('movement_variance', 0.5)
        
        # Strategy classification based on behavioral patterns
        if aggression > 0.8 and movement_var > 0.6:
            return 'berserker'
        elif aggression > 0.7 and accuracy > 0.7:
            return 'assault'
        elif aggression < 0.3 and cooperation > 0.7:
            return 'support'
        elif exploration > 0.8 and movement_var > 0.7:
            return 'scout'
        elif accuracy > 0.8 and movement_var < 0.3:
            return 'sniper'
        elif cooperation > 0.8 and aggression < 0.5:
            return 'defensive'
        elif exploration > 0.6 and aggression > 0.6:
            return 'guerrilla'
        elif all(0.3 < v < 0.7 for v in [aggression, exploration, cooperation]):
            return 'balanced'
        else:
            return 'specialist'
    
    def _extract_traits(self, agent_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key behavioral traits"""
        metrics = agent_data.get('behavioral_metrics', {})
        
        traits = {
            'aggression_level': metrics.get('avg_aggression', 0.5),
            'exploration_tendency': metrics.get('avg_exploration', 0.5),
            'cooperation_score': metrics.get('avg_cooperation', 0.5),
            'precision_rating': metrics.get('accuracy_ratio', 0.5),
            'mobility_factor': metrics.get('movement_variance', 0.5),
            'survival_instinct': min(metrics.get('survival_time', 0) / 300.0, 1.0),
            'combat_effectiveness': metrics.get('damage_efficiency', 0.5),
            'target_selectivity': metrics.get('target_preference', 0.5)
        }
        
        return traits
    
    def _detect_emergent_patterns(self, agent_data: Dict[str, Any]) -> List[str]:
        """Detect emergent behavioral patterns"""
        patterns = []
        
        action_history = agent_data.get('action_history', [])
        position_history = agent_data.get('position_history', [])
        
        if not action_history or not position_history:
            return patterns
        
        # Analyze movement patterns
        movement_pattern = self._analyze_movement_pattern(position_history)
        if movement_pattern:
            patterns.append(movement_pattern)
        
        # Analyze combat patterns
        combat_pattern = self._analyze_combat_pattern(action_history)
        if combat_pattern:
            patterns.append(combat_pattern)
        
        # Analyze adaptation patterns
        adaptation_pattern = self._analyze_adaptation_pattern(action_history)
        if adaptation_pattern:
            patterns.append(adaptation_pattern)
        
        return patterns
    
    def _analyze_movement_pattern(self, position_history: List[Tuple[float, float]]) -> Optional[str]:
        """Analyze movement patterns"""
        if len(position_history) < 20:
            return None
        
        positions = np.array(position_history[-50:])  # Last 50 positions
        
        # Calculate movement characteristics
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        avg_distance = np.mean(distances)
        distance_var = np.var(distances)
        
        # Calculate path efficiency (straight line vs actual path)
        if len(positions) > 2:
            straight_line = np.linalg.norm(positions[-1] - positions[0])
            actual_path = np.sum(distances)
            efficiency = straight_line / max(actual_path, 0.001)
        else:
            efficiency = 1.0
        
        # Classify movement pattern
        if avg_distance < 5.0:
            return 'stationary_defender'
        elif distance_var > 100.0:
            return 'erratic_mover'
        elif efficiency > 0.8:
            return 'direct_mover'
        elif efficiency < 0.3:
            return 'circling_pattern'
        elif avg_distance > 20.0:
            return 'long_range_patrol'
        else:
            return 'tactical_positioning'
    
    def _analyze_combat_pattern(self, action_history: List[Dict[str, float]]) -> Optional[str]:
        """Analyze combat decision patterns"""
        if len(action_history) < 10:
            return None
        
        recent_actions = action_history[-30:]  # Last 30 actions
        
        fire_decisions = [a['should_fire'] for a in recent_actions]
        target_preferences = [a['target_preference'] for a in recent_actions]
        
        avg_fire_rate = np.mean(fire_decisions)
        fire_consistency = 1.0 - np.var(fire_decisions)
        avg_target_pref = np.mean(target_preferences)
        
        # Classify combat pattern
        if avg_fire_rate > 0.8:
            return 'trigger_happy'
        elif avg_fire_rate < 0.2:
            return 'conservative_shooter'
        elif fire_consistency > 0.8:
            return 'consistent_combatant'
        elif avg_target_pref > 0.7:
            return 'alpha_hunter'  # Prefers strong targets
        elif avg_target_pref < 0.3:
            return 'opportunist'   # Prefers weak targets
        else:
            return 'adaptive_combatant'
    
    def _analyze_adaptation_pattern(self, action_history: List[Dict[str, float]]) -> Optional[str]:
        """Analyze how agent adapts over time"""
        if len(action_history) < 50:
            return None
        
        # Split into early and late periods
        early_actions = action_history[:len(action_history)//2]
        late_actions = action_history[len(action_history)//2:]
        
        # Calculate changes in behavior
        early_aggression = np.mean([a['should_fire'] for a in early_actions])
        late_aggression = np.mean([a['should_fire'] for a in late_actions])
        
        early_exploration = np.mean([a['exploration'] for a in early_actions])
        late_exploration = np.mean([a['exploration'] for a in late_actions])
        
        aggression_change = late_aggression - early_aggression
        exploration_change = late_exploration - early_exploration
        
        # Classify adaptation pattern
        if abs(aggression_change) > 0.3:
            if aggression_change > 0:
                return 'escalating_aggression'
            else:
                return 'learning_restraint'
        elif abs(exploration_change) > 0.3:
            if exploration_change > 0:
                return 'increasing_exploration'
            else:
                return 'territorial_settling'
        elif abs(aggression_change) < 0.1 and abs(exploration_change) < 0.1:
            return 'stable_strategy'
        else:
            return 'gradual_adaptation'
    
    def _calculate_adaptation_score(self, agent_data: Dict[str, Any]) -> float:
        """Calculate how well agent adapts to situations"""
        action_history = agent_data.get('action_history', [])
        
        if len(action_history) < 20:
            return 0.5
        
        # Measure variance in decisions over time
        decisions = ['should_fire', 'exploration', 'cooperation']
        total_variance = 0.0
        
        for decision in decisions:
            values = [a[decision] for a in action_history]
            # Higher variance indicates more adaptation
            total_variance += np.var(values)
        
        # Normalize to [0, 1] range
        adaptation_score = min(total_variance / 3.0, 1.0)
        return adaptation_score
    
    def _calculate_uniqueness_score(self, agent_data: Dict[str, Any]) -> float:
        """Calculate how unique this agent's behavior is"""
        traits = self._extract_traits(agent_data)
        
        # Calculate distance from "average" behavior (0.5 for all traits)
        average_traits = {key: 0.5 for key in traits.keys()}
        
        total_distance = 0.0
        for key in traits:
            total_distance += abs(traits[key] - average_traits[key])
        
        # Normalize by number of traits
        uniqueness_score = total_distance / len(traits)
        return min(uniqueness_score * 2.0, 1.0)  # Scale to [0, 1]

class SpeciesBehaviorAnalyzer:
    """Analyzes behavior patterns across entire species"""
    
    def __init__(self):
        self.species_data = defaultdict(list)
        self.evolution_trends = defaultdict(list)
        
    def analyze_species_evolution(self, species_id: str, generation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how a species' behavior evolves over generations"""
        
        analysis = {
            'dominant_strategies': self._find_dominant_strategies(generation_data),
            'behavioral_diversity': self._calculate_behavioral_diversity(generation_data),
            'emergent_behaviors': self._track_emergent_behaviors(generation_data),
            'strategy_stability': self._measure_strategy_stability(species_id, generation_data),
            'innovation_rate': self._calculate_innovation_rate(species_id, generation_data)
        }
        
        # Store for trend analysis
        self.species_data[species_id].append(analysis)
        
        return analysis
    
    def _find_dominant_strategies(self, generation_data: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Find the most common strategies in this generation"""
        strategy_counts = defaultdict(int)
        
        for agent_data in generation_data:
            analyzer = BehaviorAnalyzer()
            strategy = analyzer._classify_strategy(agent_data)
            strategy_counts[strategy] += 1
        
        total_agents = len(generation_data)
        strategy_percentages = [
            (strategy, count / total_agents) 
            for strategy, count in strategy_counts.items()
        ]
        
        return sorted(strategy_percentages, key=lambda x: x[1], reverse=True)
    
    def _calculate_behavioral_diversity(self, generation_data: List[Dict[str, Any]]) -> float:
        """Calculate behavioral diversity within the species"""
        if len(generation_data) < 2:
            return 0.0
        
        analyzer = BehaviorAnalyzer()
        all_traits = []
        
        for agent_data in generation_data:
            traits = analyzer._extract_traits(agent_data)
            trait_vector = list(traits.values())
            all_traits.append(trait_vector)
        
        # Calculate pairwise distances
        traits_array = np.array(all_traits)
        distances = []
        
        for i in range(len(traits_array)):
            for j in range(i + 1, len(traits_array)):
                distance = np.linalg.norm(traits_array[i] - traits_array[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _track_emergent_behaviors(self, generation_data: List[Dict[str, Any]]) -> List[str]:
        """Track new emergent behaviors in this generation"""
        analyzer = BehaviorAnalyzer()
        all_behaviors = set()
        
        for agent_data in generation_data:
            patterns = analyzer._detect_emergent_patterns(agent_data)
            all_behaviors.update(patterns)
        
        return list(all_behaviors)
    
    def _measure_strategy_stability(self, species_id: str, generation_data: List[Dict[str, Any]]) -> float:
        """Measure how stable the species' strategies are over time"""
        if species_id not in self.species_data or len(self.species_data[species_id]) < 2:
            return 0.5  # No history to compare
        
        current_strategies = self._find_dominant_strategies(generation_data)
        previous_strategies = self.species_data[species_id][-1]['dominant_strategies']
        
        # Calculate overlap in dominant strategies
        current_dict = dict(current_strategies)
        previous_dict = dict(previous_strategies)
        
        overlap = 0.0
        for strategy in current_dict:
            if strategy in previous_dict:
                overlap += min(current_dict[strategy], previous_dict[strategy])
        
        return overlap
    
    def _calculate_innovation_rate(self, species_id: str, generation_data: List[Dict[str, Any]]) -> float:
        """Calculate rate of behavioral innovation"""
        current_behaviors = set(self._track_emergent_behaviors(generation_data))
        
        if species_id not in self.species_data or len(self.species_data[species_id]) == 0:
            return len(current_behaviors) / max(len(generation_data), 1)
        
        previous_behaviors = set(self.species_data[species_id][-1]['emergent_behaviors'])
        new_behaviors = current_behaviors - previous_behaviors
        
        return len(new_behaviors) / max(len(generation_data), 1)

def create_behavior_report(species_analyses: Dict[str, Dict[str, Any]]) -> str:
    """Create a human-readable behavior analysis report"""
    
    report = "🧬 GENETIC EVOLUTION BEHAVIOR ANALYSIS REPORT\n"
    report += "=" * 60 + "\n\n"
    
    for species_id, analysis in species_analyses.items():
        report += f"📊 SPECIES: {species_id.upper()}\n"
        report += "-" * 30 + "\n"
        
        # Dominant strategies
        report += "🎯 Dominant Strategies:\n"
        for strategy, percentage in analysis['dominant_strategies'][:3]:
            report += f"   • {strategy}: {percentage:.1%}\n"
        
        # Behavioral diversity
        diversity = analysis['behavioral_diversity']
        report += f"\n🌈 Behavioral Diversity: {diversity:.3f}\n"
        
        # Emergent behaviors
        if analysis['emergent_behaviors']:
            report += "\n🚀 Emergent Behaviors:\n"
            for behavior in analysis['emergent_behaviors']:
                report += f"   • {behavior.replace('_', ' ').title()}\n"
        
        # Strategy stability
        stability = analysis['strategy_stability']
        report += f"\n⚖️  Strategy Stability: {stability:.3f}\n"
        
        # Innovation rate
        innovation = analysis['innovation_rate']
        report += f"💡 Innovation Rate: {innovation:.3f}\n"
        
        report += "\n" + "=" * 60 + "\n\n"
    
    return report

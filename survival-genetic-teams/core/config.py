"""
Configuration settings for the Multi-Agent Genetic Team Survival System
"""

class Config:
    """Global configuration for the survival simulation"""
    
    # Environment Settings
    WORLD_WIDTH = 800
    WORLD_HEIGHT = 600
    EPISODE_LENGTH = 500  # Shorter episodes for faster action
    MAX_AGENTS = 100
    
    # Team Settings
    INITIAL_TEAMS = 4  # Fewer teams for more interaction
    MIN_TEAM_SIZE = 3
    MAX_TEAM_SIZE = 8
    STARTING_TEAM_SIZE = 5  # Larger starting teams
    SPLIT_THRESHOLD = 8  # Split team when it reaches this size
    MIN_SURVIVORS_TO_CONTINUE = 2  # Minimum survivors needed to continue to next episode
    
    # Agent Settings
    AGENT_SPEED = 8.0  # Much faster movement
    AGENT_HEALTH = 60.0  # Lower health for faster combat
    AGENT_VISION_RANGE = 120.0  # Wider vision for more interaction
    AGENT_ATTACK_RANGE = 25.0  # Longer attack range
    AGENT_ATTACK_DAMAGE = 35.0  # Higher damage for faster combat
    
    # Learning Settings
    LEARNING_RATE = 0.001
    POLICY_SHARING_STRENGTH = 0.3  # How much teammates influence each other
    MUTATION_RATE = 0.1
    MUTATION_STRENGTH = 0.05
    
    # Evolution Settings
    SURVIVAL_THRESHOLD = 0.3  # Minimum survival rate to avoid team shrinking
    GROWTH_THRESHOLD = 0.7   # Survival rate needed for team growth
    ELIMINATION_GENERATIONS = 3  # Generations of poor performance before elimination
    
    # Visualization Settings
    FPS = 30
    WEB_PORT = 5002
    UPDATE_FREQUENCY = 10  # Updates per second for web interface
    SIMULATION_SPEED = 0.001  # Delay between simulation steps (seconds)
    
    # Colors for teams (will cycle through these)
    TEAM_COLORS = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#96CEB4',  # Green
        '#FFEAA7',  # Yellow
        '#DDA0DD',  # Plum
        '#98D8C8',  # Mint
        '#F7DC6F',  # Light Yellow
        '#BB8FCE',  # Light Purple
        '#85C1E9'   # Light Blue
    ]
    
    @classmethod
    def get_team_color(cls, team_id):
        """Get color for a team based on its ID"""
        return cls.TEAM_COLORS[team_id % len(cls.TEAM_COLORS)]
    
    def to_dict(self):
        """Convert config to dictionary for web interface"""
        return {
            'WORLD_WIDTH': self.WORLD_WIDTH,
            'WORLD_HEIGHT': self.WORLD_HEIGHT,
            'EPISODE_LENGTH': self.EPISODE_LENGTH,
            'MAX_AGENTS': self.MAX_AGENTS,
            'INITIAL_TEAMS': self.INITIAL_TEAMS,
            'MIN_TEAM_SIZE': self.MIN_TEAM_SIZE,
            'MAX_TEAM_SIZE': self.MAX_TEAM_SIZE,
            'STARTING_TEAM_SIZE': self.STARTING_TEAM_SIZE,
            'AGENT_SPEED': self.AGENT_SPEED,
            'AGENT_HEALTH': self.AGENT_HEALTH,
            'AGENT_VISION_RANGE': self.AGENT_VISION_RANGE,
            'AGENT_ATTACK_RANGE': self.AGENT_ATTACK_RANGE,
            'LEARNING_RATE': self.LEARNING_RATE,
            'MUTATION_RATE': self.MUTATION_RATE,
            'SIMULATION_SPEED': self.SIMULATION_SPEED,
            'FPS': self.FPS
        }
    
    def update_from_dict(self, params):
        """Update config from dictionary (for web interface)"""
        # Direct mapping for parameters that match exactly
        direct_params = [
            'EPISODE_LENGTH', 'SIMULATION_SPEED', 'INITIAL_TEAMS', 
            'STARTING_TEAM_SIZE', 'MAX_TEAM_SIZE', 'MUTATION_RATE'
        ]
        
        for param in direct_params:
            if param in params:
                setattr(self, param, params[param])
        
        # Legacy mapping for backwards compatibility
        mapping = {
            'initial_teams': 'INITIAL_TEAMS',
            'min_team_size': 'MIN_TEAM_SIZE',
            'max_team_size': 'MAX_TEAM_SIZE',
            'starting_team_size': 'STARTING_TEAM_SIZE',
            'agent_speed': 'AGENT_SPEED',
            'learning_rate': 'LEARNING_RATE',
            'mutation_rate': 'MUTATION_RATE',
            'policy_sharing_strength': 'POLICY_SHARING_STRENGTH',
            'episode_length': 'EPISODE_LENGTH',
            'simulation_speed': 'SIMULATION_SPEED'
        }
        
        for key, attr in mapping.items():
            if key in params:
                setattr(self, attr, params[key])

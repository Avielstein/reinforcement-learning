"""
Unified Web Interface for Hybrid Multi-Agent Curiosity Arena
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import logging
from typing import Tuple

from config.base_config import HybridArenaConfig


def create_app(config: HybridArenaConfig) -> Tuple[Flask, SocketIO]:
    """
    Create Flask application with SocketIO
    
    Args:
        config: Global configuration
        
    Returns:
        Tuple of (Flask app, SocketIO instance)
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hybrid-curiosity-arena-secret-key'
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    @app.route('/')
    def index():
        """Main interface page"""
        # Create a simplified config dict for template rendering
        # Extract agent composition values properly
        from config.base_config import AgentType
        template_config = {
            'environment': {
                'world_width': config.environment.world_width,
                'world_height': config.environment.world_height,
                'observation_dim': config.environment.observation_dim,
                'max_food_items': config.environment.max_food_items,
                'food_spawn_rate': config.environment.food_spawn_rate
            },
            'training': {
                'total_agents': config.training.total_agents,
                'agent_composition': {
                    'curious': config.training.agent_composition.get(AgentType.CURIOUS, 2),
                    'competitive': config.training.agent_composition.get(AgentType.COMPETITIVE, 2),
                    'hybrid': config.training.agent_composition.get(AgentType.HYBRID, 2),
                    'adaptive': config.training.agent_composition.get(AgentType.ADAPTIVE, 2)
                }
            }
        }
        return render_template('index.html', config=template_config)
    
    @app.route('/api/config')
    def get_config():
        """Get current configuration"""
        return jsonify({
            'environment': {
                'world_width': config.environment.world_width,
                'world_height': config.environment.world_height,
                'total_agents': config.training.total_agents,
                'agent_composition': {k.value: v for k, v in config.training.agent_composition.items()}
            },
            'web': {
                'port': config.web.port,
                'update_frequency': config.web.update_frequency
            }
        })
    
    @app.route('/api/status')
    def get_status():
        """Get system status"""
        return jsonify({
            'status': 'running',
            'message': 'Hybrid Multi-Agent Curiosity Arena is operational',
            'version': '1.0.0'
        })
    
    @app.route('/api/control/start', methods=['POST'])
    def start_simulation():
        """Start the simulation"""
        return jsonify({'status': 'success', 'message': 'Simulation started'})
    
    @app.route('/api/control/pause', methods=['POST'])
    def pause_simulation():
        """Pause the simulation"""
        return jsonify({'status': 'success', 'message': 'Simulation paused'})
    
    @app.route('/api/control/stop', methods=['POST'])
    def stop_simulation():
        """Stop the simulation"""
        return jsonify({'status': 'success', 'message': 'Simulation stopped'})
    
    @app.route('/api/control/reset', methods=['POST'])
    def reset_simulation():
        """Reset the simulation"""
        return jsonify({'status': 'success', 'message': 'Simulation reset'})
    
    @app.route('/api/control/start_training', methods=['POST'])
    def start_training():
        """Start training mode"""
        return jsonify({'status': 'success', 'message': 'Training started'})
    
    @app.route('/api/control/stop_training', methods=['POST'])
    def stop_training():
        """Stop training mode"""
        return jsonify({'status': 'success', 'message': 'Training stopped'})
    
    @app.route('/api/speed', methods=['POST'])
    def set_speed():
        """Set simulation speed"""
        data = request.get_json()
        multiplier = data.get('multiplier', 1.0)
        return jsonify({'status': 'success', 'speed': multiplier})
    
    @app.route('/api/population', methods=['POST'])
    def update_population():
        """Update population composition"""
        data = request.get_json()
        return jsonify({'status': 'success', 'message': 'Population updated', 'data': data})
    
    @app.route('/api/experiment/<experiment_type>', methods=['POST'])
    def load_experiment(experiment_type):
        """Load experiment preset"""
        experiments = {
            'pure_curious': {'curious': 8, 'competitive': 0, 'hybrid': 0, 'adaptive': 0},
            'pure_competitive': {'curious': 0, 'competitive': 8, 'hybrid': 0, 'adaptive': 0},
            'mixed_pop': {'curious': 2, 'competitive': 2, 'hybrid': 2, 'adaptive': 2},
            'hybrid_evolution': {'curious': 0, 'competitive': 0, 'hybrid': 4, 'adaptive': 4}
        }
        
        if experiment_type in experiments:
            return jsonify({
                'status': 'success', 
                'message': f'Loaded {experiment_type} experiment',
                'agent_composition': experiments[experiment_type]
            })
        else:
            return jsonify({'status': 'error', 'message': 'Unknown experiment type'})
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info("Client connected")
        emit('status', {'message': 'Connected to Hybrid Arena'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info("Client disconnected")
    
    return app, socketio


# Template directory is already created with the proper academic-style template

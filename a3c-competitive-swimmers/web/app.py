from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import json
import numpy as np
import sys
import os

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.competitive_waterworld import CompetitiveWaterworld
from agent.a3c_agent import A3CManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a3c-competitive-swimmers-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class A3CCompetitiveServer:
    """Server managing the A3C competitive swimmers simulation and training."""
    
    def __init__(self):
        # Environment factory
        def env_factory():
            return CompetitiveWaterworld(
                num_agents=4,
                world_width=400,
                world_height=400,
                max_food_items=20,
                max_poison_items=20,
                max_obstacles=0,  # No gray obstacles
                food_spawn_rate=0.03,
                poison_spawn_rate=0.03,
                competitive_rewards=True
            )
        
        self.env_factory = env_factory
        
        # Create demo environment for visualization
        self.demo_env = env_factory()
        self.demo_states = self.demo_env.reset()
        
        # Get environment dimensions
        test_env = env_factory()
        self.state_dim, self.action_dim = test_env.get_state_action_dims()
        
        # Create A3C manager
        self.a3c_manager = A3CManager(
            num_workers=4,
            env_factory=env_factory,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=3e-4,
            trust_region_coef=0.01
        )
        
        # Simulation state
        self.running = False
        self.training = False
        self.speed_multiplier = 1.0
        self.step_count = 0
        
        # Configuration
        self.config = {
            'learning_rate': 3e-4,
            'trust_region_coef': 0.01,
            'num_workers': 4,
            'food_spawn_rate': 0.02,
            'max_food_items': 8,
            'competitive_rewards': True
        }
        
        # Performance tracking
        self.training_metrics = {
            'steps': [],
            'rewards': [],
            'kl_divergences': [],
            'update_success_rates': [],
            'timestamps': []
        }
        
        # Threading
        self.simulation_thread = None
        self.should_stop = False
        
    def start_simulation(self):
        """Start the simulation loop."""
        if not self.running:
            self.running = True
            self.should_stop = False
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            return True
        return False
    
    def stop_simulation(self):
        """Stop the simulation loop."""
        self.running = False
        self.should_stop = True
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
        return True
    
    def reset_simulation(self):
        """Reset the simulation."""
        self.stop_simulation()
        self.demo_states = self.demo_env.reset()
        self.step_count = 0
        return True
    
    def start_training(self):
        """Start training mode."""
        if not self.training:
            self.training = True
            self.a3c_manager.start_training()
        return True
    
    def stop_training(self):
        """Stop training mode."""
        if self.training:
            self.training = False
            self.a3c_manager.stop_training()
        return True
    
    def set_speed(self, multiplier):
        """Set simulation speed multiplier."""
        self.speed_multiplier = max(0.1, min(100.0, multiplier))
        return True
    
    def update_config(self, new_config):
        """Update configuration parameters."""
        self.config.update(new_config)
        # Update environment parameters
        if 'food_spawn_rate' in new_config:
            self.demo_env.food_spawn_rate = float(new_config['food_spawn_rate'])
        if 'max_food_items' in new_config:
            self.demo_env.max_food_items = int(new_config['max_food_items'])
        if 'competitive_rewards' in new_config:
            self.demo_env.competitive_rewards = bool(new_config['competitive_rewards'])
        return True
    
    def save_model(self):
        """Save current model."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"models/a3c_competitive_swimmers_{timestamp}.pt"
        os.makedirs("models", exist_ok=True)
        self.a3c_manager.save_model(filepath)
        return filepath
    
    def load_model(self, filepath):
        """Load a pretrained model."""
        self.a3c_manager.load_model(filepath)
        return True
    
    def _simulation_loop(self):
        """Main simulation loop running in separate thread."""
        while not self.should_stop:
            if self.running:
                # Get actions from trained policy or random if not trained
                actions = []
                for i in range(self.demo_env.num_agents):
                    if self.training and len(self.a3c_manager.workers) > 0:
                        # Use trained policy
                        worker = self.a3c_manager.workers[i % len(self.a3c_manager.workers)]
                        action, _, _, _ = worker.act(self.demo_states[i])
                        actions.append(action)
                    else:
                        # Random action
                        actions.append(np.random.randint(0, 4))
                
                # Step environment
                self.demo_states, rewards, dones, info = self.demo_env.step(actions)
                self.step_count += 1
                
                # Reset if needed (though our environment doesn't terminate)
                if any(dones):
                    self.demo_states = self.demo_env.reset()
                    self.step_count = 0
                
                # Emit state update
                self._emit_state_update()
                
                # Control simulation speed - much faster
                time.sleep(0.02 / self.speed_multiplier)  # Base 50 FPS
            else:
                time.sleep(0.05)  # Pause mode
    
    def _emit_state_update(self):
        """Emit current state to connected clients."""
        # Get render state
        render_state = self.demo_env.render_state()
        
        # Get training metrics if training
        training_data = {}
        if self.training:
            global_stats = self.a3c_manager.get_global_statistics()
            worker_stats = [w.get_statistics() for w in self.a3c_manager.workers]
            
            training_data = {
                'global_stats': global_stats,
                'worker_stats': worker_stats
            }
        
        # Compile update data
        update_data = {
            'simulation': render_state,
            'step_count': self.step_count,
            'training_data': training_data,
            'config': self.config,
            'status': {
                'running': self.running,
                'training': self.training,
                'speed': self.speed_multiplier
            }
        }
        
        # Convert numpy types to JSON-serializable types
        update_data = convert_numpy_types(update_data)
        socketio.emit('state_update', update_data)

# Global server instance
server = A3CCompetitiveServer()

@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')

@app.route('/api/control/<action>', methods=['POST'])
def control_simulation(action):
    """Control simulation (start, stop, reset)."""
    try:
        if action == 'start':
            success = server.start_simulation()
        elif action == 'stop':
            success = server.stop_simulation()
        elif action == 'reset':
            success = server.reset_simulation()
        elif action == 'start_training':
            success = server.start_training()
        elif action == 'stop_training':
            success = server.stop_training()
        else:
            return jsonify({'success': False, 'error': 'Unknown action'})
        
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/speed', methods=['POST'])
def set_speed():
    """Set simulation speed."""
    try:
        data = request.get_json()
        multiplier = float(data.get('multiplier', 1.0))
        success = server.set_speed(multiplier)
        return jsonify({'success': success, 'speed': server.speed_multiplier})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration."""
    try:
        new_config = request.get_json()
        success = server.update_config(new_config)
        return jsonify({'success': success, 'config': server.config})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model/save', methods=['POST'])
def save_model():
    """Save current model."""
    try:
        filepath = server.save_model()
        return jsonify({'success': True, 'filepath': filepath})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load a model file."""
    try:
        data = request.get_json()
        filepath = data.get('filepath', '')
        success = server.load_model(filepath)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current simulation status."""
    try:
        return jsonify({
            'running': server.running,
            'training': server.training,
            'speed': server.speed_multiplier,
            'step_count': server.step_count,
            'config': server.config
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {'status': 'Connected to A3C Competitive Swimmers server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

if __name__ == '__main__':
    print("Starting A3C Competitive Swimmers server...")
    print("Open your browser to http://localhost:8080")
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)

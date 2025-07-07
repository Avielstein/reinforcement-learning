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

from environment import FishWaterworld
from agent import PPOCuriousAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'curious-fish-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class WaterworldServer:
    """Server managing the waterworld simulation and training."""
    
    def __init__(self):
        # Environment and agent
        self.env = FishWaterworld()
        self.agent = PPOCuriousAgent()
        
        # Simulation state
        self.running = False
        self.training = False
        self.speed_multiplier = 1.0
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Current state
        self.current_state = None
        self.current_action = None
        
        # Configuration (like Karpathy's spec)
        self.config = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'entropy_coef': 0.01,
            'curiosity_weight': 0.1,
            'batch_size': 64,
            'n_epochs': 10,
            'hidden_dim': 256
        }
        
        # Performance tracking
        self.reward_history = []
        self.curiosity_history = []
        self.performance_window = 1000  # Track last 1000 steps
        
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
        self.training = False
        self.should_stop = True
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
        return True
    
    def pause_simulation(self):
        """Pause the simulation."""
        self.running = False
        return True
    
    def reset_simulation(self):
        """Reset the simulation."""
        self.stop_simulation()
        self.env.reset()
        self.agent.reset_episode()
        self.step_count = 0
        self.episode_reward = 0.0
        self.current_state = None
        self.current_action = None
        return True
    
    def set_speed(self, multiplier):
        """Set simulation speed multiplier."""
        self.speed_multiplier = max(0.1, min(100.0, multiplier))
        return True
    
    def start_training(self):
        """Start training mode."""
        self.training = True
        if not self.running:
            self.start_simulation()
        return True
    
    def stop_training(self):
        """Stop training mode."""
        self.training = False
        return True
    
    def update_config(self, new_config):
        """Update configuration parameters."""
        self.config.update(new_config)
        self.agent.update_config(self.config)
        return True
    
    def load_model(self, model_data):
        """Load a pretrained model."""
        # This would handle model loading from uploaded file
        # For now, just return success
        return True
    
    def save_model(self):
        """Save current model."""
        filepath = f"models/curious_fish_{int(time.time())}.pt"
        os.makedirs("models", exist_ok=True)
        self.agent.save(filepath)
        return filepath
    
    def _simulation_loop(self):
        """Main simulation loop running in separate thread."""
        self.current_state = self.env.reset()
        
        while not self.should_stop:
            if self.running:
                # Get action from agent
                action, log_prob, value = self.agent.get_action(self.current_state, self.training)
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition if training
                if self.training:
                    self.agent.store_transition(
                        self.current_state, action, reward, next_state, done
                    )
                    
                    # Update agent if memory is full
                    if self.agent.memory.is_full():
                        metrics = self.agent.update()
                        if metrics:
                            # Emit training metrics (convert numpy types)
                            metrics = convert_numpy_types(metrics)
                            socketio.emit('training_metrics', metrics)
                
                # Update state
                self.current_state = next_state
                self.current_action = action
                self.step_count += 1
                self.episode_reward += reward
                
                # Track performance
                self.reward_history.append(reward)
                if len(self.reward_history) > self.performance_window:
                    self.reward_history.pop(0)
                
                # Get curiosity reward for tracking
                if hasattr(info, 'intrinsic_reward'):
                    self.curiosity_history.append(info['intrinsic_reward'])
                    if len(self.curiosity_history) > self.performance_window:
                        self.curiosity_history.pop(0)
                
                # Emit state update
                self._emit_state_update(reward, info)
                
                # Control simulation speed
                time.sleep(0.016 / self.speed_multiplier)  # Base 60 FPS
            else:
                time.sleep(0.1)  # Pause mode
    
    def _emit_state_update(self, reward, info):
        """Emit current state to connected clients."""
        # Get visualization data
        viz_data = self.env.get_visualization_data()
        
        # Get agent statistics
        agent_stats = self.agent.get_statistics()
        
        # Compile update data
        update_data = {
            'simulation': viz_data,
            'reward': float(reward),
            'episode_reward': float(self.episode_reward),
            'step_count': self.step_count,
            'agent_stats': agent_stats,
            'performance': {
                'recent_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
                'recent_curiosity': np.mean(self.curiosity_history[-100:]) if self.curiosity_history else 0.0,
                'reward_history': self.reward_history[-200:],  # Last 200 steps for chart
                'curiosity_history': self.curiosity_history[-200:]
            },
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
server = WaterworldServer()

@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')

@app.route('/api/control/<action>', methods=['POST'])
def control_simulation(action):
    """Control simulation (start, stop, pause, reset)."""
    try:
        if action == 'start':
            success = server.start_simulation()
        elif action == 'stop':
            success = server.stop_simulation()
        elif action == 'pause':
            success = server.pause_simulation()
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
    """Update agent configuration."""
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
        # Handle file upload (simplified for now)
        success = server.load_model(None)
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
            'episode_reward': server.episode_reward,
            'config': server.config
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {'status': 'Connected to Curious Fish server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

if __name__ == '__main__':
    print("Starting PPO Curious Fish server...")
    print("Open your browser to http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

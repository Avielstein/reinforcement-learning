"""
Flask web server for dot-follow RL interface
Handles all AI logic in Python, serves simple web interface
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import torch
import json
import os
import threading
import time
from datetime import datetime

# Import our existing utilities
from dot_follow_environment import DotFollowEnv
from dot_follow_trainer import DotFollowLearner

app = Flask(__name__)

class WebInterface:
    def __init__(self):
        self.env = DotFollowEnv('circular')
        self.learner = None
        self.model_loaded = False
        self.is_running = False
        self.is_paused = False
        
        # Simulation state
        self.episode = 0
        self.step_count = 0
        self.current_reward = 0.0
        self.total_reward = 0.0
        
        # Parameters that can be adjusted via web interface
        self.params = {
            'target_speed': 8.0,
            'target_radius': 20.0,
            'exploration_noise': 0.1,
            'action_scale': 1.0,
            'current_strength': 2.0
        }
        
        # History for charts
        self.reward_history = []
        self.distance_history = []
        self.speed_history = []
        
        # Threading
        self.simulation_thread = None
        self.stop_simulation = False
        
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            self.learner = DotFollowLearner(self.env.movement_pattern)
            self.learner.load_model(model_path)
            self.learner.load_best()
            self.model_loaded = True
            return True, f"Model loaded successfully from {model_path}"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def export_model_weights(self):
        """Export model weights as JSON for web interface"""
        if not self.model_loaded:
            return None
        
        try:
            # Get model state dict and convert to JSON-serializable format
            state_dict = self.learner.ac.state_dict()
            json_weights = {}
            
            for key, tensor in state_dict.items():
                json_weights[key] = tensor.cpu().numpy().tolist()
            
            return json_weights
        except Exception as e:
            print(f"Error exporting weights: {e}")
            return None
    
    def get_fish_action(self):
        """Get action from loaded model or random if no model"""
        if not self.model_loaded:
            # Random action
            return np.random.uniform(-1, 1, 2)
        
        try:
            obs = self.env._obs()
            action = self.learner.ac.act(torch.as_tensor(obs, dtype=torch.float32))
            
            # Add exploration noise if specified
            if self.params['exploration_noise'] > 0:
                noise = np.random.normal(0, self.params['exploration_noise'], 2)
                action = action + noise
            
            # Scale action
            action = action * self.params['action_scale']
            
            # Clip to valid range
            action = np.clip(action, -1, 1)
            
            return action
        except Exception as e:
            print(f"Error getting action: {e}")
            return np.random.uniform(-1, 1, 2)
    
    def update_environment_params(self):
        """Update environment parameters based on web interface settings"""
        self.env.target.speed = self.params['target_speed']
        self.env.target.radius = self.params['target_radius']
        
        # Update current strength
        for current in self.env.currents:
            current.strength = self.params['current_strength']
    
    def simulation_step(self):
        """Single simulation step"""
        if not self.is_running or self.is_paused:
            return
        
        # Update environment parameters
        self.update_environment_params()
        
        # Get action from model
        action = self.get_fish_action()
        
        # Take step in environment
        obs, reward, done, info = self.env.step(action)
        
        # Update state
        self.step_count += 1
        self.current_reward = reward
        self.total_reward += reward
        
        # Calculate metrics
        target_distance = np.linalg.norm(self.env.position - self.env.target.position)
        fish_speed = np.linalg.norm(self.env.velocity)
        
        # Update history (keep last 200 points)
        self.reward_history.append(reward)
        self.distance_history.append(target_distance)
        self.speed_history.append(fish_speed)
        
        if len(self.reward_history) > 200:
            self.reward_history.pop(0)
            self.distance_history.pop(0)
            self.speed_history.pop(0)
        
        # Reset if episode done
        if done:
            self.env.reset()
            self.episode += 1
            self.step_count = 0
            self.total_reward = 0.0
    
    def run_simulation(self):
        """Main simulation loop"""
        while not self.stop_simulation:
            if self.is_running and not self.is_paused:
                self.simulation_step()
            time.sleep(0.05)  # 20 FPS
    
    def start_simulation(self):
        """Start the simulation"""
        if not self.simulation_thread or not self.simulation_thread.is_alive():
            self.stop_simulation = False
            self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.simulation_thread.start()
        
        self.is_running = True
        self.is_paused = False
    
    def pause_simulation(self):
        """Pause/unpause simulation"""
        self.is_paused = not self.is_paused
    
    def stop_simulation_thread(self):
        """Stop simulation"""
        self.is_running = False
        self.stop_simulation = True
    
    def reset_environment(self):
        """Reset the environment"""
        self.env.reset()
        self.episode += 1
        self.step_count = 0
        self.current_reward = 0.0
        self.total_reward = 0.0
    
    def set_movement_pattern(self, pattern):
        """Change target movement pattern"""
        self.env.set_movement_pattern(pattern)
        self.env.target.time = 0  # Reset pattern
    
    def get_state(self):
        """Get current simulation state for web interface"""
        return {
            'fish': {
                'x': float(self.env.position[0]),
                'y': float(self.env.position[1]),
                'vx': float(self.env.velocity[0]),
                'vy': float(self.env.velocity[1])
            },
            'target': {
                'x': float(self.env.target.position[0]),
                'y': float(self.env.target.position[1])
            },
            'currents': [
                {
                    'x': float(current.position[0]),
                    'y': float(current.position[1]),
                    'direction': float(np.arctan2(current.direction[1], current.direction[0])),
                    'strength': float(current.strength),
                    'radius': float(current.radius)
                }
                for current in self.env.currents
            ],
            'stats': {
                'episode': self.episode,
                'step_count': self.step_count,
                'current_reward': round(self.current_reward, 3),
                'total_reward': round(self.total_reward, 3),
                'target_distance': round(np.linalg.norm(self.env.position - self.env.target.position), 2),
                'fish_speed': round(np.linalg.norm(self.env.velocity), 2),
                'pattern': self.env.movement_pattern,
                'model_loaded': self.model_loaded,
                'is_running': self.is_running,
                'is_paused': self.is_paused
            },
            'history': {
                'rewards': self.reward_history[-50:],  # Last 50 points for chart
                'distances': self.distance_history[-50:],
                'speeds': self.speed_history[-50:]
            },
            'params': self.params
        }

# Global interface instance
interface = WebInterface()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    """Get current simulation state"""
    return jsonify(interface.get_state())

@app.route('/api/control/<action>', methods=['POST'])
def control_simulation(action):
    """Control simulation (start/pause/stop/reset)"""
    if action == 'start':
        interface.start_simulation()
        return jsonify({'success': True, 'message': 'Simulation started'})
    elif action == 'pause':
        interface.pause_simulation()
        return jsonify({'success': True, 'message': 'Simulation paused/unpaused'})
    elif action == 'stop':
        interface.stop_simulation_thread()
        return jsonify({'success': True, 'message': 'Simulation stopped'})
    elif action == 'reset':
        interface.reset_environment()
        return jsonify({'success': True, 'message': 'Environment reset'})
    else:
        return jsonify({'success': False, 'message': 'Unknown action'})

@app.route('/api/pattern/<pattern>', methods=['POST'])
def set_pattern(pattern):
    """Set movement pattern"""
    valid_patterns = ['circular', 'figure8', 'random_walk', 'zigzag', 'spiral']
    if pattern in valid_patterns:
        interface.set_movement_pattern(pattern)
        return jsonify({'success': True, 'message': f'Pattern set to {pattern}'})
    else:
        return jsonify({'success': False, 'message': 'Invalid pattern'})

@app.route('/api/params', methods=['POST'])
def update_params():
    """Update simulation parameters"""
    data = request.json
    for key, value in data.items():
        if key in interface.params:
            interface.params[key] = float(value)
    
    return jsonify({'success': True, 'message': 'Parameters updated', 'params': interface.params})

@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load a model file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    # Save uploaded file temporarily
    filename = f"temp_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    
    # Load the model
    success, message = interface.load_model(filepath)
    
    # Clean up temp file
    try:
        os.remove(filepath)
    except:
        pass
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/model/weights')
def get_model_weights():
    """Get model weights as JSON"""
    weights = interface.export_model_weights()
    if weights:
        return jsonify({'success': True, 'weights': weights})
    else:
        return jsonify({'success': False, 'message': 'No model loaded or export failed'})

@app.route('/models/<filename>')
def serve_model(filename):
    """Serve model files"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    print("Starting Dot Follow RL Web Interface...")
    print("Navigate to http://localhost:5001 to access the interface")
    print("\nTo use:")
    print("1. Train a model using Python (train_dot_follow.ipynb or simple_demo.py)")
    print("2. Load the .pt model file via the web interface")
    print("3. Adjust parameters and watch the fish behavior")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

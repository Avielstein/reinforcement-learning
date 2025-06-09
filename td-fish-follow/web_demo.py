#!/usr/bin/env python3
"""
Working web interface for TD Fish Follow.
"""

import sys
import os
import json
import time
import threading
import numpy as np
import torch
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Import the working training components
from train import SimpleFishEnvironment, SimpleTDAgent


class TDFishWebInterface:
    """Web interface for TD fish learning."""
    
    def __init__(self, port: int = 5001):
        self.app = Flask(__name__, 
                        template_folder='web',
                        static_folder='web/static',
                        static_url_path='/static')
        self.port = port
        
        # Training state
        self.env = None
        self.agent = None
        self.training_active = False
        self.current_episode = 0
        self.training_stats = []
        
        # Current simulation state
        self.current_state = {
            'fish_position': [400, 300],
            'target_position': [400, 300],
            'episode_reward': 0,
            'step_count': 0,
            'distance_to_target': 0,
            'td_error': 0,
            'action': [0, 0],
            'learning': True
        }
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/start_training', methods=['POST'])
        def start_training():
            data = request.json
            pattern = data.get('pattern', 'random_walk')
            td_method = data.get('td_method', 'td_lambda')
            
            # Create environment and agent with REAL TD learning
            self.env = SimpleFishEnvironment(pattern)
            self.agent = SimpleTDAgent(method=td_method)
            
            # Start training in background thread
            self.training_active = True
            self.current_episode = 0
            self.training_stats = []
            
            training_thread = threading.Thread(target=self._training_loop)
            training_thread.daemon = True
            training_thread.start()
            
            return jsonify({
                'status': 'started', 
                'pattern': pattern, 
                'method': td_method,
                'learning': True,
                'message': f'Real {td_method.upper()} learning started - fish will learn from scratch!'
            })
        
        @self.app.route('/api/stop_training', methods=['POST'])
        def stop_training():
            self.training_active = False
            return jsonify({'status': 'stopped'})
        
        @self.app.route('/api/current_state')
        def get_current_state():
            return jsonify(self.current_state)
        
        @self.app.route('/api/training_stats')
        def get_training_stats():
            return jsonify({
                'episode': self.current_episode,
                'stats': self.training_stats[-50:] if self.training_stats else []
            })
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset_simulation():
            if self.env and self.agent:
                obs = self.env.reset()
                self.agent.prev_action = None
                self._update_current_state()
            return jsonify({'status': 'reset'})
        
        @self.app.route('/api/step', methods=['POST'])
        def manual_step():
            if not self.env or not self.agent:
                return jsonify({'error': 'No environment loaded'})
            
            # Get action from REAL agent
            obs = self.env._get_observation()
            action = self.agent.select_action(obs, deterministic=True)
            
            # Take step
            next_obs, reward, done, info = self.env.step(action)
            
            # Update state
            self._update_current_state(action, reward, info)
            
            if done:
                obs = self.env.reset()
                self.agent.prev_action = None
            
            return jsonify(self.current_state)
        
        @self.app.route('/api/load_model', methods=['POST'])
        def load_model():
            data = request.json
            model_path = data.get('model_path')
            
            if not model_path or not os.path.exists(model_path):
                return jsonify({'error': 'Model file not found'})
            
            try:
                # Create and load agent
                self.agent = SimpleTDAgent()
                self.agent.load(model_path)
                self.agent.epsilon = 0  # No exploration for demo
                
                # Create environment
                pattern = data.get('pattern', 'circular')
                self.env = SimpleFishEnvironment(pattern)
                
                self.current_state['learning'] = False
                
                return jsonify({
                    'status': 'loaded',
                    'model_path': model_path,
                    'message': 'Pre-trained model loaded - fish is already trained!'
                })
                
            except Exception as e:
                return jsonify({'error': f'Failed to load model: {str(e)}'})
    
    def _training_loop(self):
        """REAL TD learning training loop."""
        print("🧠 Starting REAL TD learning - fish will be random at first!")
        
        while self.training_active and self.env and self.agent:
            try:
                # Run one episode
                obs = self.env.reset()
                self.agent.prev_action = None
                
                episode_reward = 0
                episode_distances = []
                episode_td_errors = []
                step_count = 0
                done = False
                
                while not done and self.training_active:
                    # Select action using REAL agent
                    action = self.agent.select_action(obs)
                    
                    # Take step
                    next_obs, reward, done, info = self.env.step(action)
                    
                    # REAL TD learning update
                    stats = self.agent.update(obs, action, reward, next_obs, done)
                    
                    # Track metrics
                    episode_reward += reward
                    episode_distances.append(info['distance_to_target'])
                    episode_td_errors.append(stats['td_error'])
                    
                    # Update current state for visualization
                    self._update_current_state(action, reward, info, stats)
                    
                    obs = next_obs
                    step_count += 1
                    
                    # Visualization delay
                    time.sleep(0.05)
                
                # End episode with REAL learning
                self.agent.end_episode()
                avg_distance = np.mean(episode_distances)
                
                # Store episode data
                self.current_episode += 1
                episode_stats = {
                    'episode': self.current_episode,
                    'reward': episode_reward,
                    'avg_distance': avg_distance,
                    'steps': step_count,
                    'avg_td_error': np.mean(episode_td_errors),
                    'exploration_rate': self.agent.epsilon
                }
                
                self.training_stats.append(episode_stats)
                
                # Print learning progress
                if self.current_episode <= 10 or self.current_episode % 5 == 0:
                    print(f"Episode {self.current_episode}: "
                          f"Reward={episode_reward:.1f}, "
                          f"Avg Distance={avg_distance:.1f}, "
                          f"TD Error={np.mean(episode_td_errors):.4f}, "
                          f"Exploration={self.agent.epsilon:.3f}")
                
                # Show learning milestones
                if self.current_episode == 1:
                    print("🎯 Episode 1: Fish is moving randomly - this is normal!")
                elif self.current_episode == 10:
                    print("🧠 Episode 10: Fish should start showing some learning...")
                elif self.current_episode == 25:
                    print("📈 Episode 25: Fish should be getting better at following!")
                
            except Exception as e:
                print(f"Training error: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def _update_current_state(self, action=None, reward=None, info=None, stats=None):
        """Update current state for visualization."""
        if self.env:
            self.current_state.update({
                'fish_position': self.env.fish_pos.tolist(),
                'target_position': self.env.target_pos.tolist(),
                'step_count': self.env.step_count,
                'tank_width': self.env.tank_width,
                'tank_height': self.env.tank_height,
                'fish_size': 8,
                'target_size': 6,
                'learning': True
            })
            
            if action is not None:
                self.current_state['action'] = action.tolist()
            
            if reward is not None:
                self.current_state['episode_reward'] = self.current_state.get('episode_reward', 0) + reward
            
            if info is not None:
                self.current_state['distance_to_target'] = info['distance_to_target']
            
            if stats and 'td_error' in stats:
                self.current_state['td_error'] = stats['td_error']
            
            # Add learning indicators
            if self.agent:
                self.current_state['exploration_rate'] = self.agent.epsilon
    
    def run(self, debug=False):
        """Run the web interface."""
        print(f"🌐 Starting TD Fish Follow Web Interface on http://localhost:{self.port}")
        print("This interface uses REAL TD learning - the fish will start random and learn!")
        print("Watch the fish improve over episodes as it learns the task.")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)


def main():
    """Main entry point."""
    print("🐠 TD Fish Follow - Web Interface")
    print("=" * 50)
    print("This interface demonstrates REAL TD learning!")
    print("The fish starts with random behavior and learns over time.")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser to: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        interface = TDFishWebInterface(port=5001)
        interface.run()
    except KeyboardInterrupt:
        print("\n👋 Shutting down web server...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Flask web server for the Multi-Agent Genetic Team Survival System
"""

from flask import Flask, render_template, jsonify, request
import threading
import time
import json
from typing import Dict, Any

from core.config import Config
from simulation.episode_runner import EpisodeRunner

class WebInterface:
    """Web interface for the survival simulation"""
    
    def __init__(self):
        self.app = Flask(__name__, template_folder='../visualization/web_interface')
        self.config = Config()
        self.runner = None
        self.simulation_thread = None
        self.is_running = False
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/start', methods=['POST'])
        def start_simulation():
            if self.runner is None:
                self.runner = EpisodeRunner(self.config)
            
            if not self.is_running:
                self.is_running = True
                self.runner.start_simulation(background=True)
                return jsonify({'status': 'started', 'message': 'Simulation started'})
            else:
                return jsonify({'status': 'error', 'message': 'Simulation already running'})
        
        @self.app.route('/api/pause', methods=['POST'])
        def pause_simulation():
            if self.runner and self.is_running:
                self.runner.pause_simulation()
                return jsonify({'status': 'paused', 'message': 'Simulation paused'})
            else:
                return jsonify({'status': 'error', 'message': 'No simulation running'})
        
        @self.app.route('/api/resume', methods=['POST'])
        def resume_simulation():
            if self.runner:
                self.runner.resume_simulation()
                return jsonify({'status': 'resumed', 'message': 'Simulation resumed'})
            else:
                return jsonify({'status': 'error', 'message': 'No simulation to resume'})
        
        @self.app.route('/api/stop', methods=['POST'])
        def stop_simulation():
            if self.runner and self.is_running:
                self.runner.stop_simulation()
                self.is_running = False
                return jsonify({'status': 'stopped', 'message': 'Simulation stopped'})
            else:
                return jsonify({'status': 'error', 'message': 'No simulation running'})
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset_simulation():
            if self.runner:
                self.runner.stop_simulation()
                self.is_running = False
            
            self.runner = EpisodeRunner(self.config)
            return jsonify({'status': 'reset', 'message': 'Simulation reset'})
        
        @self.app.route('/api/state')
        def get_state():
            if self.runner:
                try:
                    state = self.runner.get_real_time_stats()
                    return jsonify(state)
                except Exception as e:
                    return jsonify({'error': str(e)})
            else:
                return jsonify({
                    'episode': 0,
                    'step': 0,
                    'is_running': False,
                    'is_paused': False,
                    'environment': {'alive_agents': 0, 'total_agents': 0},
                    'population': {'total_teams': 0, 'total_agents': 0}
                })
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def handle_config():
            if request.method == 'GET':
                return jsonify(self.config.to_dict())
            else:
                try:
                    new_params = request.json
                    self.config.update_from_dict(new_params)
                    if self.runner:
                        self.runner.adjust_config(new_params)
                    return jsonify({'status': 'updated', 'config': self.config.to_dict()})
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/performance')
        def get_performance():
            if self.runner:
                try:
                    performance = self.runner.get_performance_summary()
                    return jsonify(performance)
                except Exception as e:
                    return jsonify({'error': str(e)})
            else:
                return jsonify({'error': 'No simulation running'})
    
    def run(self, host='localhost', port=5002, debug=False):
        """Run the web server"""
        print(f"🌐 Starting web interface at http://{host}:{port}")
        print("🧬 Multi-Agent Genetic Team Survival System")
        print("=" * 50)
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point for web interface"""
    web_interface = WebInterface()
    web_interface.run()

if __name__ == '__main__':
    main()

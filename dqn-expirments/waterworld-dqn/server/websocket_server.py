"""WebSocket server for real-time communication with research interface."""

import os
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import threading
import time

from .data_manager import DataManager
from .api_handlers import setup_handlers

def create_app():
    """Create and configure Flask application with SocketIO."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'waterworld-research-key'
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Setup API handlers
    setup_handlers(socketio, data_manager)
    
    @app.route('/')
    def index():
        """Serve the main research interface."""
        # Read the HTML template
        html_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'index.html')
        try:
            with open(html_path, 'r') as f:
                html_content = f.read()
            return html_content
        except FileNotFoundError:
            # Fallback minimal HTML if file not found
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>WaterWorld DQN Research Interface</title>
                <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
            </head>
            <body>
                <h1>WaterWorld DQN Research Interface</h1>
                <p>Interface files not found. Please ensure static/index.html exists.</p>
                <div id="status">Connecting...</div>
                <script>
                    const socket = io();
                    socket.on('connect', () => {
                        document.getElementById('status').textContent = 'Connected to server';
                    });
                </script>
            </body>
            </html>
            """)
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return {'status': 'healthy', 'service': 'waterworld-dqn'}
    
    # Store references for access from handlers
    app.socketio = socketio
    app.data_manager = data_manager
    
    return app

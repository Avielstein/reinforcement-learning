#!/usr/bin/env python3
"""
Main entry point for WaterWorld Double DQN research implementation.
"""

import sys
import os
import subprocess

def install_requirements():
    """Install required packages if needed."""
    try:
        import numpy
        import flask
        import flask_socketio
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")

def start_server():
    """Start the research interface server."""
    from server.websocket_server import create_app
    from config.ui_config import UIConfig
    
    print("🔬 WaterWorld Double DQN Research Interface")
    print("=" * 50)
    print(f"🌐 Open your browser to: http://localhost:{UIConfig.PORT}")
    print("📊 Research-grade RL training visualization")
    print("⚙️  Real-time parameter adjustment")
    print("🧠 Double DQN algorithm with sensing")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    app = create_app()
    app.run(
        host=UIConfig.HOST,
        port=UIConfig.PORT,
        debug=UIConfig.DEBUG
    )

def main():
    """Main entry point."""
    # Check if we're in the right directory
    if not os.path.exists("config"):
        print("❌ Please run this script from the waterworld-dqn directory.")
        return
    
    # Install dependencies if needed
    install_requirements()
    
    # Start the server
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()

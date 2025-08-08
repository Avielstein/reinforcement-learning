#!/usr/bin/env python3
"""
PPO + Curiosity Fish: Run on port 9000 to avoid conflict
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the web application
from web.app import app, socketio

if __name__ == '__main__':
    print("🐟 PPO + Curiosity Fish: Interactive Waterworld RL")
    print("=" * 50)
    print("Starting server...")
    print("Open your browser to: http://localhost:9000")
    print("=" * 50)
    print()
    print("Features:")
    print("- Real-time PPO + Curiosity training")
    print("- 152D sensor system (30 rays + proprioception)")
    print("- Live parameter tuning (Karpathy-style)")
    print("- Interactive visualization")
    print("- Model save/load functionality")
    print()
    print("Controls:")
    print("- Edit parameters in the left panel")
    print("- Use speed controls: 'Go very fast', 'Go fast', etc.")
    print("- Start/stop training and simulation")
    print("- Watch the fish learn in real-time!")
    print()
    
    try:
        socketio.run(app, host='0.0.0.0', port=9000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Make sure you have installed all requirements:")
        print("pip install -r requirements.txt")

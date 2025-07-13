#!/usr/bin/env python3
"""
A3C Competitive Swimmers: Interactive Multi-Agent RL

Main entry point for the application.
Run this to start the web server and begin training/testing the competitive swimmers.
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the web application
from web.app import app, socketio

if __name__ == '__main__':
    print("🏊‍♂️ A3C Competitive Swimmers: Interactive Multi-Agent RL")
    print("=" * 60)
    print("Starting server...")
    print("Open your browser to: http://localhost:8080")
    print("=" * 60)
    print()
    print("Features:")
    print("- Real-time A3C + Trust Region training")
    print("- Multi-agent competitive learning")
    print("- 152D sensor system (30 rays + proprioception)")
    print("- Live parameter tuning")
    print("- Interactive visualization")
    print("- Model save/load functionality")
    print()
    print("Controls:")
    print("- Start/stop training and simulation")
    print("- Adjust speed: 'Go Slow', 'Normal', 'Go Fast', 'Very Fast'")
    print("- Edit environment parameters")
    print("- Watch agents learn to compete in real-time!")
    print()
    
    try:
        socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Make sure you have installed all requirements:")
        print("pip install -r requirements.txt")

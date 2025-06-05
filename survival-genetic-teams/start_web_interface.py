"""
Startup script for the Multi-Agent Genetic Team Survival System Web Interface
"""

import sys
import os

# Add the current directory to the path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from simulation.web_server import main

if __name__ == '__main__':
    print("🧬 Multi-Agent Genetic Team Survival System - Web Interface")
    print("=" * 60)
    print("🌐 Starting web server...")
    print("📱 Open your browser and navigate to: http://localhost:5002")
    print("🎮 Use the web interface to control the simulation")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Web server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting web server: {e}")

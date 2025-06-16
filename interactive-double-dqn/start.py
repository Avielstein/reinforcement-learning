#!/usr/bin/env python3
"""
Quick start script for the Double DQN Observatory
Installs dependencies and starts the server
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_server():
    """Start the observatory server"""
    print("🚀 Starting Double DQN Observatory...")
    print("🌐 Open your browser to: http://localhost:8080")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "server.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except FileNotFoundError:
        print("❌ server.py not found. Make sure you're in the correct directory.")

def main():
    print("🐠 Double DQN Observatory - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("server.py"):
        print("❌ server.py not found. Please run this script from the interactive-double-dqn directory.")
        return
    
    if not os.path.exists("index.html"):
        print("❌ index.html not found. Please run this script from the interactive-double-dqn directory.")
        return
    
    # Install dependencies
    if install_requirements():
        print()
        start_server()
    else:
        print("❌ Cannot start server due to dependency installation failure.")

if __name__ == "__main__":
    main()

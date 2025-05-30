#!/usr/bin/env python3
"""
Startup script for the Dot Follow RL Web Interface
"""

import os
import sys
import subprocess

def main():
    print("🐠 Dot Follow RL Web Interface")
    print("=" * 50)
    
    # Check if we have a trained model
    model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    
    if model_files:
        print(f"✅ Found trained models: {', '.join(model_files)}")
    else:
        print("⚠️  No trained models found (.pt files)")
        print("   Run simple_demo.py first to train a model:")
        print("   python simple_demo.py")
        print()
        
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting...")
            return
    
    print()
    print("🚀 Starting web server...")
    print("📱 Open your browser to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Start the Flask server
        from web_server import app
        app.run(debug=False, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the tank-sim/dot-follow directory")
        print("2. Check that all required packages are installed:")
        print("   pip install flask torch numpy matplotlib")
        print("3. Verify the Python environment has access to the utils module")

if __name__ == "__main__":
    main()

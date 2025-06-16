#!/usr/bin/env python3
"""
Simple test server for the Double DQN Observatory UI
Provides mock data and WebSocket communication for testing the interface
"""

import json
import time
import math
import random
import threading
from flask import Flask, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dqn-observatory-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for mock training
class MockTrainingState:
    def __init__(self):
        self.is_training = False
        self.episode = 0
        self.step = 0
        self.fish_x = 300
        self.fish_y = 200
        self.target_x = 300
        self.target_y = 200
        self.reward = 0.0
        self.cumulative_reward = 0.0
        self.epsilon = 1.0
        self.avg_q_value = 0.0
        self.loss = 0.0
        self.steps_since_target_update = 0
        self.episode_rewards = []
        
        # Parameters
        self.learning_rate = 0.001
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.batch_size = 32

training_state = MockTrainingState()

@app.route('/')
def index():
    """Serve the main HTML file"""
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "index.html not found. Make sure it's in the same directory as server.py", 404

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files if needed"""
    return send_from_directory('.', filename)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('training_status', {'status': 'idle'})
    
    # Send initial state
    emit('training_update', {
        'episode': training_state.episode,
        'step': training_state.step,
        'reward': training_state.cumulative_reward,
        'distance': calculate_distance_to_center(),
        'avg_reward': calculate_avg_reward(),
        'epsilon': training_state.epsilon,
        'avg_q_value': training_state.avg_q_value,
        'loss': training_state.loss,
        'steps_since_target_update': training_state.steps_since_target_update,
        'fish_position': {'x': training_state.fish_x, 'y': training_state.fish_y}
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_training')
def handle_start_training():
    """Start mock training"""
    print('Starting training...')
    training_state.is_training = True
    emit('training_status', {'status': 'training'})
    
    # Start training thread
    if not hasattr(handle_start_training, 'thread_running'):
        handle_start_training.thread_running = True
        training_thread = threading.Thread(target=mock_training_loop)
        training_thread.daemon = True
        training_thread.start()

@socketio.on('pause_training')
def handle_pause_training():
    """Pause mock training"""
    print('Pausing training...')
    training_state.is_training = False
    emit('training_status', {'status': 'paused'})

@socketio.on('reset_training')
def handle_reset_training():
    """Reset training state"""
    print('Resetting training...')
    training_state.is_training = False
    training_state.episode = 0
    training_state.step = 0
    training_state.fish_x = 300
    training_state.fish_y = 200
    training_state.cumulative_reward = 0.0
    training_state.epsilon = training_state.epsilon_start
    training_state.episode_rewards = []
    
    emit('training_status', {'status': 'reset'})
    emit('training_update', {
        'episode': 0,
        'step': 0,
        'reward': 0.0,
        'distance': calculate_distance_to_center(),
        'avg_reward': 0.0,
        'epsilon': training_state.epsilon,
        'avg_q_value': 0.0,
        'loss': 0.0,
        'steps_since_target_update': 0,
        'fish_position': {'x': training_state.fish_x, 'y': training_state.fish_y}
    })

@socketio.on('save_model')
def handle_save_model():
    """Mock save model"""
    print('Saving model...')
    emit('parameter_confirmation', {'message': 'Model saved successfully!'})

@socketio.on('parameter_update')
def handle_parameter_update(data):
    """Handle parameter updates from UI"""
    param = data.get('parameter')
    value = data.get('value')
    
    print(f'Parameter update: {param} = {value}')
    
    # Update the parameter
    if param == 'learning_rate':
        training_state.learning_rate = value
    elif param == 'epsilon_start':
        training_state.epsilon_start = value
        if not training_state.is_training:  # Only update current epsilon if not training
            training_state.epsilon = value
    elif param == 'epsilon_decay':
        training_state.epsilon_decay = value
    elif param == 'target_update_freq':
        training_state.target_update_freq = int(value)
    elif param == 'batch_size':
        training_state.batch_size = int(value)
    
    emit('parameter_confirmation', {'parameter': param, 'value': value})

def calculate_distance_to_center():
    """Calculate distance from fish to center"""
    center_x, center_y = 300, 200
    dx = training_state.fish_x - center_x
    dy = training_state.fish_y - center_y
    return math.sqrt(dx*dx + dy*dy)

def calculate_avg_reward():
    """Calculate average reward over recent episodes"""
    if not training_state.episode_rewards:
        return 0.0
    recent_rewards = training_state.episode_rewards[-10:]  # Last 10 episodes
    return sum(recent_rewards) / len(recent_rewards)

def mock_training_loop():
    """Mock training loop that simulates DQN learning"""
    print('Mock training loop started')
    
    while True:
        if not training_state.is_training:
            time.sleep(0.1)
            continue
            
        # Simulate one training step
        training_state.step += 1
        training_state.steps_since_target_update += 1
        
        # Mock fish movement (gradually move toward center with some randomness)
        center_x, center_y = 300, 200
        dx = center_x - training_state.fish_x
        dy = center_y - training_state.fish_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Add some learning progress - fish gets better over time
        learning_progress = min(training_state.episode / 100.0, 1.0)  # Progress over 100 episodes
        
        # Movement toward center with decreasing randomness as learning progresses
        move_strength = 2.0 + learning_progress * 3.0  # Stronger movement as it learns
        random_strength = 3.0 * (1.0 - learning_progress)  # Less randomness as it learns
        
        if distance > 1:
            # Move toward center
            training_state.fish_x += (dx / distance) * move_strength + random.uniform(-random_strength, random_strength)
            training_state.fish_y += (dy / distance) * move_strength + random.uniform(-random_strength, random_strength)
        
        # Keep fish in bounds
        training_state.fish_x = max(20, min(580, training_state.fish_x))
        training_state.fish_y = max(20, min(380, training_state.fish_y))
        
        # Calculate reward (higher when closer to center)
        current_distance = calculate_distance_to_center()
        max_distance = math.sqrt(300*300 + 200*200)  # Max possible distance
        step_reward = 1.0 - (current_distance / max_distance)
        
        # Bonus for being very close
        if current_distance < 20:
            step_reward += 2.0
        elif current_distance < 50:
            step_reward += 1.0
        
        training_state.cumulative_reward += step_reward
        
        # Update epsilon (exploration decay)
        training_state.epsilon = max(0.01, training_state.epsilon * training_state.epsilon_decay)
        
        # Mock Q-values and loss
        training_state.avg_q_value = random.uniform(0, 20) * (1 + learning_progress)
        training_state.loss = random.uniform(0.001, 0.1) * (1 - learning_progress * 0.8)
        
        # Target network update
        if training_state.steps_since_target_update >= training_state.target_update_freq:
            training_state.steps_since_target_update = 0
        
        # End episode after 200 steps or if very close to center
        episode_end = (training_state.step % 200 == 0) or (current_distance < 10)
        
        if episode_end:
            training_state.episode += 1
            training_state.episode_rewards.append(training_state.cumulative_reward)
            training_state.cumulative_reward = 0.0
            training_state.step = 0
            
            # Reset fish position for new episode
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(100, 200)
            training_state.fish_x = 300 + radius * math.cos(angle)
            training_state.fish_y = 200 + radius * math.sin(angle)
        
        # Send update to client
        socketio.emit('training_update', {
            'episode': training_state.episode,
            'step': training_state.step,
            'reward': training_state.cumulative_reward,
            'distance': current_distance,
            'avg_reward': calculate_avg_reward(),
            'epsilon': training_state.epsilon,
            'avg_q_value': training_state.avg_q_value,
            'loss': training_state.loss,
            'steps_since_target_update': training_state.steps_since_target_update,
            'fish_position': {'x': training_state.fish_x, 'y': training_state.fish_y}
        })
        
        # Control update frequency (10 FPS)
        time.sleep(0.1)

if __name__ == '__main__':
    print("🐠 Double DQN Observatory Server Starting...")
    print("Open your browser to: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    
    try:
        socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\nServer stopped.")

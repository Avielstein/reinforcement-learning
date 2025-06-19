"""API handlers for WebSocket communication."""

import threading
import time
from flask_socketio import emit

def setup_handlers(socketio, data_manager):
    """Setup WebSocket event handlers."""
    
    training_thread = None
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print("🔗 Client connected")
        # Send initial state
        emit('training_update', data_manager.get_current_state())
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print("🔌 Client disconnected")
    
    @socketio.on('start_training')
    def handle_start_training():
        """Start training simulation."""
        nonlocal training_thread
        
        data_manager.start_training()
        
        # Start training loop in background thread
        if training_thread is None or not training_thread.is_alive():
            training_thread = threading.Thread(
                target=start_training_loop,
                args=(socketio, data_manager),
                daemon=True
            )
            training_thread.start()
        
        emit('training_status', {'status': 'training'})
    
    @socketio.on('pause_training')
    def handle_pause_training():
        """Pause training simulation."""
        data_manager.pause_training()
        emit('training_status', {'status': 'paused'})
    
    @socketio.on('reset_training')
    def handle_reset_training():
        """Reset training to initial state."""
        data_manager.reset_training()
        emit('training_status', {'status': 'reset'})
        emit('training_update', data_manager.get_current_state())
    
    @socketio.on('save_model')
    def handle_save_model():
        """Save current model."""
        filename = data_manager.save_model()
        emit('model_saved', {'filename': filename})
    
    @socketio.on('parameter_update')
    def handle_parameter_update(data):
        """Update training parameter."""
        parameter = data.get('parameter')
        value = data.get('value')
        
        if parameter and value is not None:
            data_manager.update_parameter(parameter, value)
            emit('parameter_confirmation', {
                'parameter': parameter,
                'value': value
            })
    
    @socketio.on('get_state')
    def handle_get_state():
        """Get current training state."""
        emit('training_update', data_manager.get_current_state())
    
    @socketio.on('get_available_models')
    def handle_get_available_models():
        """Get list of available trained models."""
        models = data_manager.get_available_models()
        emit('available_models', {'models': models})
    
    @socketio.on('load_model')
    def handle_load_model(data):
        """Load a trained model."""
        model_path = data.get('model_path')
        if model_path:
            success, message = data_manager.load_model(model_path)
            emit('model_loaded', {
                'success': success,
                'message': message,
                'model_path': model_path
            })
            if success:
                emit('training_update', data_manager.get_current_state())
    
    @socketio.on('toggle_real_training')
    def handle_toggle_real_training(data):
        """Toggle between mock and real DQN training."""
        use_real_dqn = data.get('use_real_dqn', False)
        data_manager.set_real_training_mode(use_real_dqn)
        emit('training_mode_changed', {
            'use_real_dqn': use_real_dqn,
            'message': 'Real DQN training enabled' if use_real_dqn else 'Mock training enabled'
        })


def start_training_loop(socketio, data_manager):
    """Background training loop for mock DQN training."""
    print("🧠 Starting mock training loop...")
    
    while data_manager.is_training:
        try:
            # Simulate training step
            training_data = data_manager.step()
            
            # Emit training update
            socketio.emit('training_update', training_data)
            
            # Control update frequency
            time.sleep(1.0 / 10)  # 10 Hz updates
            
        except Exception as e:
            print(f"Training loop error: {e}")
            break
    
    print("🛑 Training loop stopped")

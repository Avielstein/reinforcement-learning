#!/usr/bin/env python3
"""
Real-Time CFL Learning Monitor
Tracks how CFL discovers causal relationships over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import time
from collections import deque
import threading
import queue

class CFLLearningMonitor:
    """Monitor CFL learning process in real-time"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        
        # Data storage
        self.causal_history = deque(maxlen=max_history)
        self.variance_history = deque(maxlen=max_history)
        self.stability_history = deque(maxlen=max_history)
        self.epoch_history = deque(maxlen=max_history)
        
        # Change detection
        self.last_causal_state = None
        self.stability_threshold = 0.001
        self.change_alerts = []
        
        # Setup plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('CFL Learning Monitor - Real-Time Causal Discovery', fontsize=16)
        
        # Initialize plots
        self.setup_plots()
        
    def setup_plots(self):
        """Initialize all monitoring plots"""
        
        # Plot 1: Real-time causal effects
        self.axes[0, 0].set_title('Live Causal Effects by Dimension')
        self.axes[0, 0].set_xlabel('Training Epoch')
        self.axes[0, 0].set_ylabel('Causal Effect Strength')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].legend(['Background', 'Paddles', 'Walls', 'Ball'])
        
        # Plot 2: Variance evolution (object discovery)
        self.axes[0, 1].set_title('Object Discovery (Variance Over Time)')
        self.axes[0, 1].set_xlabel('Training Epoch')
        self.axes[0, 1].set_ylabel('Causal Variance')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning stability
        self.axes[1, 0].set_title('Learning Stability')
        self.axes[1, 0].set_xlabel('Training Epoch')
        self.axes[1, 0].set_ylabel('Stability Score')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].axhline(y=self.stability_threshold, color='red', 
                               linestyle='--', label='Stable Threshold')
        
        # Plot 4: Change detection alerts
        self.axes[1, 1].set_title('Causal Relationship Changes')
        self.axes[1, 1].set_xlabel('Training Epoch')
        self.axes[1, 1].set_ylabel('Change Magnitude')
        self.axes[1, 1].grid(True, alpha=0.3)
        
    def add_causal_data(self, epoch, causal_effects):
        """Add new causal effects data point"""
        
        # Store data
        self.epoch_history.append(epoch)
        self.causal_history.append(causal_effects.copy())
        
        # Calculate variance (object importance) - handle single values
        if len(causal_effects.shape) == 1:
            # Single frame average - calculate variance across dimensions
            variances = causal_effects  # Use the values themselves as variance proxy
        else:
            variances = np.var(causal_effects, axis=0)
        self.variance_history.append(variances)
        
        # Calculate stability (how much things are changing)
        if self.last_causal_state is not None:
            stability = np.mean(np.abs(causal_effects - self.last_causal_state))
            self.stability_history.append(stability)
            
            # Detect significant changes
            if stability > self.stability_threshold * 10:  # Major change threshold
                self.change_alerts.append((epoch, stability))
                print(f"🚨 MAJOR CAUSAL CHANGE detected at epoch {epoch}! Magnitude: {stability:.6f}")
                
        self.last_causal_state = causal_effects.copy()
        
    def update_plots(self):
        """Update all plots with latest data"""
        
        if len(self.causal_history) < 2:
            return
            
        epochs = list(self.epoch_history)
        
        # Clear all plots
        for ax in self.axes.flat:
            ax.clear()
        
        self.setup_plots()
        
        # Plot 1: Live causal effects
        causal_data = np.array(list(self.causal_history))
        colors = ['gray', 'blue', 'red', 'orange']
        labels = ['Causal Type 1', 'Causal Type 2', 'Causal Type 3', 'Causal Type 4']
        
        for i in range(min(4, causal_data.shape[1])):
            self.axes[0, 0].plot(epochs, causal_data[:, i], 
                               color=colors[i], label=labels[i], alpha=0.8)
        self.axes[0, 0].legend()
        
        # Plot 2: Variance evolution
        variance_data = np.array(list(self.variance_history))
        for i in range(min(4, variance_data.shape[1])):
            self.axes[0, 1].plot(epochs, variance_data[:, i], 
                               color=colors[i], label=labels[i], alpha=0.8)
        self.axes[0, 1].legend()
        
        # Plot 3: Stability
        if len(self.stability_history) > 0:
            self.axes[1, 0].plot(epochs[1:], list(self.stability_history), 
                               color='green', linewidth=2)
            self.axes[1, 0].axhline(y=self.stability_threshold, color='red', 
                                   linestyle='--', label='Stable Threshold')
            self.axes[1, 0].legend()
        
        # Plot 4: Change alerts
        if self.change_alerts:
            alert_epochs, alert_magnitudes = zip(*self.change_alerts)
            self.axes[1, 1].scatter(alert_epochs, alert_magnitudes, 
                                  color='red', s=100, alpha=0.7, label='Major Changes')
            self.axes[1, 1].legend()
        
        plt.tight_layout()
        
    def analyze_learning_progress(self):
        """Analyze current learning state"""
        
        if len(self.variance_history) < 10:
            return "Insufficient data for analysis"
            
        current_variances = list(self.variance_history)[-1]
        recent_stability = np.mean(list(self.stability_history)[-10:]) if len(self.stability_history) >= 10 else float('inf')
        
        # Determine learning state
        analysis = []
        analysis.append("🧠 CFL Learning Analysis:")
        analysis.append("=" * 30)
        
        # Object discovery status
        most_important = np.argmax(current_variances)
        object_names = ['Background', 'Paddles', 'Walls', 'Ball']
        analysis.append(f"Most important object: {object_names[most_important]} (variance: {current_variances[most_important]:.8f})")
        
        # Learning stability
        if recent_stability < self.stability_threshold:
            analysis.append("✅ Learning is STABLE - CFL has figured out the causal structure")
        else:
            analysis.append("🔄 Still LEARNING - causal relationships are changing")
            
        # Change detection
        recent_changes = len([alert for alert in self.change_alerts if alert[0] > max(self.epoch_history) - 50])
        if recent_changes > 0:
            analysis.append(f"⚠️  {recent_changes} major changes detected in last 50 epochs")
        else:
            analysis.append("🎯 No major changes recently - consistent learning")
            
        return "\n".join(analysis)

def monitor_existing_training():
    """Monitor CFL training from saved results"""
    
    print("🔍 CFL Learning Monitor - Analyzing Existing Training")
    print("=" * 60)
    
    # Load existing training data
    pickle_path = "cfl_results/experiment0002/dataset_train/CondDensityEstimator_results.pickle"
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        pyx = data['pyx']  # (5000, 4) causal effects
        print(f"✅ Loaded {pyx.shape[0]} frames with {pyx.shape[1]} causal dimensions")
        
        # Create monitor
        monitor = CFLLearningMonitor()
        
        # Simulate training progression by feeding data in chunks
        chunk_size = 500  # Frames per "epoch"
        num_epochs = pyx.shape[0] // chunk_size
        
        print(f"📊 Simulating {num_epochs} training epochs...")
        
        for epoch in range(num_epochs):
            start_idx = epoch * chunk_size
            end_idx = min((epoch + 1) * chunk_size, pyx.shape[0])
            
            # Get causal effects for this "epoch"
            epoch_data = pyx[start_idx:end_idx]
            avg_causal_effects = np.mean(epoch_data, axis=0)
            
            # Add to monitor
            monitor.add_causal_data(epoch, avg_causal_effects)
            
            # Update plots every few epochs
            if epoch % 2 == 0:
                monitor.update_plots()
                plt.pause(0.1)  # Brief pause for animation effect
        
        # Final analysis
        print("\n" + monitor.analyze_learning_progress())
        
        # Show final plots
        monitor.update_plots()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def create_live_monitor():
    """Create a live monitor for new CFL training"""
    
    print("🚀 Setting up Live CFL Monitor...")
    print("This would hook into active CFL training to show real-time learning")
    print("For now, run monitor_existing_training() to see how it works!")
    
    # This would integrate with the actual CFL training loop
    # by modifying train_cfl_pong.py to call monitor.add_causal_data()
    # at each epoch

if __name__ == "__main__":
    print("🎮 CFL Learning Monitor")
    print("=" * 30)
    print("1. Analyzing existing Pong training...")
    
    monitor_existing_training()

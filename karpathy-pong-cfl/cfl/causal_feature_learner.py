"""
Causal Feature Learning (CFL) Implementation
Based on the CFL research for discovering macrovariables that preserve causal relationships
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import pickle
import os

class CFLEncoder(nn.Module):
    """Neural network encoder for learning feature representations"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, feature_dim: int = 64):
        super(CFLEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.Tanh()  # Bounded output
        )
        
    def forward(self, x):
        return self.encoder(x)

class CFLPredictor(nn.Module):
    """Neural network for predicting effects from causes"""
    
    def __init__(self, cause_dim: int, effect_dim: int, hidden_dim: int = 128):
        super(CFLPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(cause_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, effect_dim)
        )
        
    def forward(self, x):
        return self.predictor(x)

class CausalFeatureLearner:
    """
    Causal Feature Learning for discovering macrovariables in RL environments
    
    This implementation focuses on Pong pixels -> game outcomes causal relationships
    """
    
    def __init__(self, 
                 input_dim: int = 6400,  # 80x80 Pong pixels
                 n_macro_causes: int = 16,  # Number of cause macrovariables
                 n_macro_effects: int = 4,   # Number of effect macrovariables
                 feature_dim: int = 64,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu',
                 max_data_size: int = 15000):  # Limit data to ~10 episodes worth
        
        self.input_dim = input_dim
        self.n_macro_causes = n_macro_causes
        self.n_macro_effects = n_macro_effects
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.max_data_size = max_data_size
        
        # Neural networks
        self.cause_encoder = CFLEncoder(input_dim, feature_dim=feature_dim).to(self.device)
        self.effect_encoder = CFLEncoder(3, feature_dim=16).to(self.device)  # reward, done, action
        self.predictor = CFLPredictor(feature_dim, 16).to(self.device)
        
        # Optimizers
        self.cause_optimizer = optim.Adam(self.cause_encoder.parameters(), lr=learning_rate)
        self.effect_optimizer = optim.Adam(self.effect_encoder.parameters(), lr=learning_rate)
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        # Clustering models
        self.cause_clusterer = KMeans(n_clusters=n_macro_causes, random_state=42)
        self.effect_clusterer = KMeans(n_clusters=n_macro_effects, random_state=42)
        
        # Data storage with sliding window
        self.cause_data = []  # Raw pixel observations
        self.effect_data = []  # Game outcomes (reward, done, action)
        self.cause_features = []  # Encoded cause features
        self.effect_features = []  # Encoded effect features
        
        # Macrovariable assignments
        self.cause_macro_labels = None
        self.effect_macro_labels = None
        
        # Training statistics
        self.training_losses = []
        self.is_trained = False
        
    def add_data(self, observation: np.ndarray, reward: float, done: bool, action: int):
        """Add a data point for CFL training with sliding window"""
        # Store cause data (pixel observation)
        self.cause_data.append(observation.flatten())
        
        # Store effect data (reward, done, action)
        effect = np.array([reward, float(done), float(action)])
        self.effect_data.append(effect)
        
        # Maintain sliding window - keep only recent data
        if len(self.cause_data) > self.max_data_size:
            # Remove oldest data points
            excess = len(self.cause_data) - self.max_data_size
            self.cause_data = self.cause_data[excess:]
            self.effect_data = self.effect_data[excess:]
        
    def encode_features(self):
        """Encode raw data into feature representations"""
        if len(self.cause_data) == 0:
            return
            
        # Convert to tensors
        cause_tensor = torch.FloatTensor(np.array(self.cause_data)).to(self.device)
        effect_tensor = torch.FloatTensor(np.array(self.effect_data)).to(self.device)
        
        # Encode features
        with torch.no_grad():
            self.cause_features = self.cause_encoder(cause_tensor).cpu().numpy()
            self.effect_features = self.effect_encoder(effect_tensor).cpu().numpy()
    
    def train_encoders(self, epochs: int = 100, batch_size: int = 64):
        """Train the encoder networks to predict effects from causes"""
        if len(self.cause_data) < batch_size:
            print(f"Not enough data for training. Need at least {batch_size} samples.")
            return
            
        cause_tensor = torch.FloatTensor(np.array(self.cause_data)).to(self.device)
        effect_tensor = torch.FloatTensor(np.array(self.effect_data)).to(self.device)
        
        dataset_size = len(self.cause_data)
        
        print(f"Training CFL encoders on {dataset_size} samples...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_causes = cause_tensor[batch_indices]
                batch_effects = effect_tensor[batch_indices]
                
                # Forward pass
                cause_features = self.cause_encoder(batch_causes)
                effect_features = self.effect_encoder(batch_effects)
                predicted_effects = self.predictor(cause_features)
                
                # Loss: prediction error + feature regularization
                prediction_loss = nn.MSELoss()(predicted_effects, effect_features)
                
                # Regularization to encourage diverse features
                cause_reg = torch.mean(torch.std(cause_features, dim=0))
                effect_reg = torch.mean(torch.std(effect_features, dim=0))
                
                total_loss = prediction_loss - 0.01 * (cause_reg + effect_reg)
                
                # Backward pass
                self.cause_optimizer.zero_grad()
                self.effect_optimizer.zero_grad()
                self.predictor_optimizer.zero_grad()
                
                total_loss.backward()
                
                self.cause_optimizer.step()
                self.effect_optimizer.step()
                self.predictor_optimizer.step()
                
                epoch_loss += total_loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
        
        print("Encoder training completed!")
        
    def discover_macrovariables(self):
        """Discover macrovariables through clustering"""
        if len(self.cause_features) == 0:
            self.encode_features()
            
        if len(self.cause_features) == 0:
            print("No features available for clustering")
            return
            
        print("Discovering macrovariables through clustering...")
        
        # Standardize features
        scaler_cause = StandardScaler()
        scaler_effect = StandardScaler()
        
        cause_features_scaled = scaler_cause.fit_transform(self.cause_features)
        effect_features_scaled = scaler_effect.fit_transform(self.effect_features)
        
        # Use simple clustering to avoid sklearn issues
        print("Using simple k-means clustering...")
        self.cause_macro_labels = self._simple_kmeans(cause_features_scaled, self.n_macro_causes)
        self.effect_macro_labels = self._simple_kmeans(effect_features_scaled, self.n_macro_effects)
        
        print(f"Discovered {self.n_macro_causes} cause macrovariables")
        print(f"Discovered {self.n_macro_effects} effect macrovariables")
        
        # Analyze macrovariables
        self._analyze_macrovariables()
        
        self.is_trained = True
    
    def _simple_kmeans(self, data, k, max_iters=10):
        """Simple k-means fallback implementation"""
        n_samples, n_features = data.shape
        
        # Initialize centers randomly
        centers = data[np.random.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to closest centers
            distances = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
            assignments = np.argmin(distances, axis=0)
            
            # Update centers
            new_centers = np.array([data[assignments == i].mean(axis=0) if np.sum(assignments == i) > 0 
                                   else centers[i] for i in range(k)])
            
            # Check convergence
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        return assignments
        
    def _analyze_macrovariables(self):
        """Analyze the discovered macrovariables"""
        print("\nMacrovariable Analysis:")
        print("-" * 40)
        
        # Cause macrovariables
        unique_causes, cause_counts = np.unique(self.cause_macro_labels, return_counts=True)
        print("Cause Macrovariables:")
        for i, (label, count) in enumerate(zip(unique_causes, cause_counts)):
            percentage = count / len(self.cause_macro_labels) * 100
            print(f"  Macro-cause {label}: {count} samples ({percentage:.1f}%)")
            
        # Effect macrovariables
        unique_effects, effect_counts = np.unique(self.effect_macro_labels, return_counts=True)
        print("\nEffect Macrovariables:")
        for i, (label, count) in enumerate(zip(unique_effects, effect_counts)):
            percentage = count / len(self.effect_macro_labels) * 100
            print(f"  Macro-effect {label}: {count} samples ({percentage:.1f}%)")
            
        # Causal relationships
        print("\nCausal Relationships (Cause -> Effect):")
        for cause_label in unique_causes:
            cause_mask = self.cause_macro_labels == cause_label
            effect_distribution = self.effect_macro_labels[cause_mask]
            unique_effects_for_cause, counts = np.unique(effect_distribution, return_counts=True)
            
            print(f"  Macro-cause {cause_label} ->", end=" ")
            for effect_label, count in zip(unique_effects_for_cause, counts):
                prob = count / len(effect_distribution)
                print(f"Effect {effect_label}({prob:.2f})", end=" ")
            print()
    
    def transform_observation(self, observation: np.ndarray) -> int:
        """Transform a raw observation to its macrovariable representation"""
        if not self.is_trained:
            raise ValueError("CFL must be trained before transforming observations")
            
        # Encode the observation
        obs_tensor = torch.FloatTensor(observation.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.cause_encoder(obs_tensor).cpu().numpy()
        
        # Standardize features (same as during training)
        scaler = StandardScaler()
        scaler.fit(self.cause_features)
        features_scaled = scaler.transform(features)
        
        # Find closest macrovariable using simple distance
        if not hasattr(self, 'cause_centers'):
            # Compute centers from training data
            self.cause_centers = np.array([self.cause_features[self.cause_macro_labels == i].mean(axis=0) 
                                         for i in range(self.n_macro_causes)])
            self.cause_centers_scaled = scaler.transform(self.cause_centers)
        
        # Find closest center
        distances = np.sqrt(((self.cause_centers_scaled - features_scaled)**2).sum(axis=1))
        macro_label = np.argmin(distances)
        return macro_label
    
    def get_macrovariable_features(self, macro_label: int) -> np.ndarray:
        """Get the representative features for a macrovariable"""
        if not self.is_trained:
            raise ValueError("CFL must be trained before getting macrovariable features")
            
        # Find all observations belonging to this macrovariable
        mask = self.cause_macro_labels == macro_label
        if not np.any(mask):
            return np.zeros(self.feature_dim)
            
        # Return mean features for this macrovariable
        return np.mean(self.cause_features[mask], axis=0)
    
    def save(self, filepath: str):
        """Save the trained CFL model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'cause_encoder_state': self.cause_encoder.state_dict(),
            'effect_encoder_state': self.effect_encoder.state_dict(),
            'predictor_state': self.predictor.state_dict(),
            'cause_clusterer': self.cause_clusterer,
            'effect_clusterer': self.effect_clusterer,
            'cause_features': self.cause_features,
            'effect_features': self.effect_features,
            'cause_macro_labels': self.cause_macro_labels,
            'effect_macro_labels': self.effect_macro_labels,
            'training_losses': self.training_losses,
            'is_trained': self.is_trained,
            'config': {
                'input_dim': self.input_dim,
                'n_macro_causes': self.n_macro_causes,
                'n_macro_effects': self.n_macro_effects,
                'feature_dim': self.feature_dim
            }
        }
        
        torch.save(save_dict, filepath)
        print(f"CFL model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained CFL model"""
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.cause_encoder.load_state_dict(save_dict['cause_encoder_state'])
        self.effect_encoder.load_state_dict(save_dict['effect_encoder_state'])
        self.predictor.load_state_dict(save_dict['predictor_state'])
        
        self.cause_clusterer = save_dict['cause_clusterer']
        self.effect_clusterer = save_dict['effect_clusterer']
        self.cause_features = save_dict['cause_features']
        self.effect_features = save_dict['effect_features']
        self.cause_macro_labels = save_dict['cause_macro_labels']
        self.effect_macro_labels = save_dict['effect_macro_labels']
        self.training_losses = save_dict['training_losses']
        self.is_trained = save_dict['is_trained']
        
        print(f"CFL model loaded from {filepath}")
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        if len(self.training_losses) == 0:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses)
        plt.title('CFL Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_macrovariables(self, save_path: str = None):
        """Visualize the discovered macrovariables"""
        if not self.is_trained:
            print("CFL must be trained before visualization")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cause feature distribution
        axes[0, 0].scatter(self.cause_features[:, 0], self.cause_features[:, 1], 
                          c=self.cause_macro_labels, cmap='tab10', alpha=0.6)
        axes[0, 0].set_title('Cause Macrovariables (Feature Space)')
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        
        # Plot 2: Effect feature distribution
        axes[0, 1].scatter(self.effect_features[:, 0], self.effect_features[:, 1], 
                          c=self.effect_macro_labels, cmap='tab10', alpha=0.6)
        axes[0, 1].set_title('Effect Macrovariables (Feature Space)')
        axes[0, 1].set_xlabel('Feature 1')
        axes[0, 1].set_ylabel('Feature 2')
        
        # Plot 3: Macrovariable distribution
        unique_causes, cause_counts = np.unique(self.cause_macro_labels, return_counts=True)
        axes[1, 0].bar(unique_causes, cause_counts)
        axes[1, 0].set_title('Cause Macrovariable Distribution')
        axes[1, 0].set_xlabel('Macrovariable ID')
        axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Training loss
        axes[1, 1].plot(self.training_losses)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

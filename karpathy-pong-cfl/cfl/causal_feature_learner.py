"""
Causal Feature Learning (CFL) for Reinforcement Learning
Redesigned to properly discover macro-states that accelerate RL learning

Key Insight: Learn causally meaningful state abstractions by grouping states
that have similar effects when the same actions are applied.

Correct Causal Structure:
- Causes (X): [current_state, action_taken]
- Effects (Y): [next_state, reward_received, value_change]
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
from collections import defaultdict

class StateEncoder(nn.Module):
    """Neural network encoder for learning state representations"""
    
    def __init__(self, input_dim: int = 6400, hidden_dim: int = 512, feature_dim: int = 64):
        super(StateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.Tanh()  # Bounded output [-1, 1]
        )
        
    def forward(self, x):
        return self.encoder(x)

class TransitionPredictor(nn.Module):
    """Neural network for predicting state transitions and rewards"""
    
    def __init__(self, state_dim: int = 64, action_dim: int = 2, hidden_dim: int = 128):
        super(TransitionPredictor, self).__init__()
        
        # Predict next state features
        self.state_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
            nn.Tanh()
        )
        
        # Predict reward
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state_features, action_onehot):
        combined = torch.cat([state_features, action_onehot], dim=1)
        next_state_pred = self.state_predictor(combined)
        reward_pred = self.reward_predictor(combined)
        return next_state_pred, reward_pred

class CausalFeatureLearner:
    """
    CFL for discovering macro-states that accelerate RL learning
    
    This implementation learns to group states that have similar causal effects
    when actions are applied, creating a compressed state representation.
    """
    
    def __init__(self, 
                 input_dim: int = 6400,  # 80x80 Pong pixels
                 n_macro_states: int = 16,  # Number of macro-states to discover
                 feature_dim: int = 64,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu',
                 max_data_size: int = 10000):
        
        self.input_dim = input_dim
        self.n_macro_states = n_macro_states
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.max_data_size = max_data_size
        
        # Neural networks
        self.state_encoder = StateEncoder(input_dim, feature_dim=feature_dim).to(self.device)
        self.transition_predictor = TransitionPredictor(feature_dim).to(self.device)
        
        # Optimizers
        self.encoder_optimizer = optim.Adam(self.state_encoder.parameters(), lr=learning_rate)
        self.predictor_optimizer = optim.Adam(self.transition_predictor.parameters(), lr=learning_rate)
        
        # Clustering model for macro-state discovery
        self.clusterer = KMeans(n_clusters=n_macro_states, random_state=42, n_init=10)
        
        # Data storage with proper causal structure
        self.transitions = []  # List of (state, action, next_state, reward) tuples
        self.state_features = []  # Encoded state features
        self.macro_state_labels = None  # Cluster assignments
        self.macro_state_centers = None  # Cluster centers in feature space
        
        # Training statistics
        self.training_losses = {'state_prediction': [], 'reward_prediction': [], 'total': []}
        self.is_trained = False
        
        # Action encoding (Pong has actions 2 and 3, we'll map to 0 and 1)
        self.action_mapping = {2: 0, 3: 1}  # UP: 0, DOWN: 1
        
    def add_transition(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float):
        """Add a state transition for CFL learning"""
        # Store transition with proper causal structure
        transition = {
            'state': state.flatten(),
            'action': self.action_mapping.get(action, 0),  # Map Pong actions to 0/1
            'next_state': next_state.flatten(),
            'reward': reward
        }
        
        self.transitions.append(transition)
        
        # Maintain sliding window to prevent memory issues
        if len(self.transitions) > self.max_data_size:
            excess = len(self.transitions) - self.max_data_size
            self.transitions = self.transitions[excess:]
    
    def encode_states(self):
        """Encode all states into feature representations"""
        if len(self.transitions) == 0:
            return
            
        # Extract states and encode them
        states = np.array([t['state'] for t in self.transitions])
        next_states = np.array([t['next_state'] for t in self.transitions])
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(states).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        
        # Encode features
        with torch.no_grad():
            state_features = self.state_encoder(state_tensor).cpu().numpy()
            next_state_features = self.state_encoder(next_state_tensor).cpu().numpy()
        
        self.state_features = state_features
        self.next_state_features = next_state_features
    
    def train_transition_model(self, epochs: int = 200, batch_size: int = 64):
        """Train the transition prediction model"""
        if len(self.transitions) < batch_size:
            print(f"Not enough data for training. Need at least {batch_size} samples.")
            return
            
        print(f"Training transition model on {len(self.transitions)} transitions...")
        
        # Prepare data
        states = np.array([t['state'] for t in self.transitions])
        actions = np.array([t['action'] for t in self.transitions])
        next_states = np.array([t['next_state'] for t in self.transitions])
        rewards = np.array([t['reward'] for t in self.transitions])
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(states).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        
        # One-hot encode actions
        action_onehot = torch.zeros(len(actions), 2).to(self.device)
        action_onehot.scatter_(1, action_tensor.unsqueeze(1), 1)
        
        dataset_size = len(self.transitions)
        
        for epoch in range(epochs):
            epoch_state_loss = 0.0
            epoch_reward_loss = 0.0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_states = state_tensor[batch_indices]
                batch_actions = action_onehot[batch_indices]
                batch_next_states = next_state_tensor[batch_indices]
                batch_rewards = reward_tensor[batch_indices]
                
                # Encode current states
                current_features = self.state_encoder(batch_states)
                target_features = self.state_encoder(batch_next_states)
                
                # Predict transitions
                pred_next_features, pred_rewards = self.transition_predictor(current_features, batch_actions)
                
                # Compute losses
                state_loss = nn.MSELoss()(pred_next_features, target_features.detach())
                reward_loss = nn.MSELoss()(pred_rewards, batch_rewards)
                
                # Total loss with regularization
                feature_reg = torch.mean(torch.std(current_features, dim=0))  # Encourage diverse features
                total_loss = state_loss + reward_loss - 0.01 * feature_reg
                
                # Backward pass
                self.encoder_optimizer.zero_grad()
                self.predictor_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.transition_predictor.parameters(), 1.0)
                
                self.encoder_optimizer.step()
                self.predictor_optimizer.step()
                
                epoch_state_loss += state_loss.item()
                epoch_reward_loss += reward_loss.item()
                n_batches += 1
            
            # Record losses
            avg_state_loss = epoch_state_loss / n_batches
            avg_reward_loss = epoch_reward_loss / n_batches
            avg_total_loss = avg_state_loss + avg_reward_loss
            
            self.training_losses['state_prediction'].append(avg_state_loss)
            self.training_losses['reward_prediction'].append(avg_reward_loss)
            self.training_losses['total'].append(avg_total_loss)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch:3d}: State Loss = {avg_state_loss:.6f}, Reward Loss = {avg_reward_loss:.6f}")
        
        print("Transition model training completed!")
        
    def discover_macro_states(self):
        """Discover macro-states through causal clustering"""
        if len(self.state_features) == 0:
            self.encode_states()
            
        if len(self.state_features) == 0:
            print("No state features available for clustering")
            return
            
        print("Discovering macro-states through causal clustering...")
        
        # Standardize features for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.state_features)
        
        # Perform clustering to discover macro-states
        try:
            self.macro_state_labels = self.clusterer.fit_predict(features_scaled)
            self.macro_state_centers = scaler.inverse_transform(self.clusterer.cluster_centers_)
            self.feature_scaler = scaler  # Store for later use
        except Exception as e:
            print(f"Clustering failed, using simple k-means: {e}")
            self.macro_state_labels = self._simple_kmeans(features_scaled, self.n_macro_states)
            # Compute centers manually
            self.macro_state_centers = np.array([
                self.state_features[self.macro_state_labels == i].mean(axis=0) 
                for i in range(self.n_macro_states)
            ])
            self.feature_scaler = scaler
        
        print(f"Discovered {self.n_macro_states} macro-states")
        
        # Analyze the discovered macro-states
        self._analyze_macro_states()
        
        self.is_trained = True
    
    def _simple_kmeans(self, data, k, max_iters=20):
        """Simple k-means fallback implementation"""
        n_samples, n_features = data.shape
        
        # Initialize centers randomly
        centers = data[np.random.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to closest centers
            distances = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
            assignments = np.argmin(distances, axis=0)
            
            # Update centers
            new_centers = np.array([
                data[assignments == i].mean(axis=0) if np.sum(assignments == i) > 0 
                else centers[i] for i in range(k)
            ])
            
            # Check convergence
            if np.allclose(centers, new_centers, rtol=1e-4):
                break
            centers = new_centers
        
        return assignments
        
    def _analyze_macro_states(self):
        """Analyze the discovered macro-states"""
        print("\nMacro-State Analysis:")
        print("-" * 50)
        
        # Distribution of macro-states
        unique_states, state_counts = np.unique(self.macro_state_labels, return_counts=True)
        print("Macro-State Distribution:")
        for state_id, count in zip(unique_states, state_counts):
            percentage = count / len(self.macro_state_labels) * 100
            print(f"  State {state_id:2d}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Analyze action-outcome relationships for each macro-state
        print("\nAction-Outcome Analysis by Macro-State:")
        for state_id in unique_states:
            state_mask = self.macro_state_labels == state_id
            state_transitions = [t for i, t in enumerate(self.transitions) if state_mask[i]]
            
            if len(state_transitions) == 0:
                continue
                
            # Analyze action distribution
            actions = [t['action'] for t in state_transitions]
            rewards = [t['reward'] for t in state_transitions]
            
            action_counts = defaultdict(int)
            action_rewards = defaultdict(list)
            
            for action, reward in zip(actions, rewards):
                action_counts[action] += 1
                action_rewards[action].append(reward)
            
            print(f"  State {state_id:2d}:", end=" ")
            for action in sorted(action_counts.keys()):
                count = action_counts[action]
                avg_reward = np.mean(action_rewards[action])
                action_name = "UP" if action == 0 else "DOWN"
                print(f"{action_name}({count}, r={avg_reward:.2f})", end=" ")
            print()
    
    def transform_state(self, state: np.ndarray) -> int:
        """Transform a raw state to its macro-state representation"""
        if not self.is_trained:
            raise ValueError("CFL must be trained before transforming states")
            
        # Encode the state
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.state_encoder(state_tensor).cpu().numpy()
        
        # Standardize features (same as during training)
        features_scaled = self.feature_scaler.transform(features)
        
        # Find closest macro-state center
        distances = np.sqrt(((self.feature_scaler.transform(self.macro_state_centers) - features_scaled)**2).sum(axis=1))
        macro_state_id = np.argmin(distances)
        
        return macro_state_id
    
    def get_macro_state_representation(self, macro_state_id: int) -> np.ndarray:
        """Get a one-hot representation of the macro-state"""
        if not self.is_trained:
            raise ValueError("CFL must be trained before getting representations")
            
        representation = np.zeros(self.n_macro_states)
        representation[macro_state_id] = 1.0
        return representation
    
    def save(self, filepath: str):
        """Save the trained CFL model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'state_encoder_state': self.state_encoder.state_dict(),
            'transition_predictor_state': self.transition_predictor.state_dict(),
            'clusterer': self.clusterer,
            'state_features': self.state_features,
            'macro_state_labels': self.macro_state_labels,
            'macro_state_centers': self.macro_state_centers,
            'feature_scaler': self.feature_scaler if hasattr(self, 'feature_scaler') else None,
            'training_losses': self.training_losses,
            'is_trained': self.is_trained,
            'action_mapping': self.action_mapping,
            'config': {
                'input_dim': self.input_dim,
                'n_macro_states': self.n_macro_states,
                'feature_dim': self.feature_dim
            }
        }
        
        torch.save(save_dict, filepath)
        print(f"CFL model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained CFL model"""
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.state_encoder.load_state_dict(save_dict['state_encoder_state'])
        self.transition_predictor.load_state_dict(save_dict['transition_predictor_state'])
        
        self.clusterer = save_dict['clusterer']
        self.state_features = save_dict['state_features']
        self.macro_state_labels = save_dict['macro_state_labels']
        self.macro_state_centers = save_dict['macro_state_centers']
        self.feature_scaler = save_dict.get('feature_scaler')
        self.training_losses = save_dict['training_losses']
        self.is_trained = save_dict['is_trained']
        self.action_mapping = save_dict.get('action_mapping', {2: 0, 3: 1})
        
        print(f"CFL model loaded from {filepath}")
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        if len(self.training_losses['total']) == 0:
            return
            
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.training_losses['state_prediction'], label='State Prediction')
        plt.plot(self.training_losses['reward_prediction'], label='Reward Prediction')
        plt.plot(self.training_losses['total'], label='Total Loss')
        plt.title('CFL Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        if hasattr(self, 'state_features') and len(self.state_features) > 0:
            plt.scatter(self.state_features[:, 0], self.state_features[:, 1], 
                       c=self.macro_state_labels, cmap='tab20', alpha=0.6, s=10)
            plt.title('Macro-States in Feature Space')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.colorbar(label='Macro-State ID')
        
        plt.subplot(1, 3, 3)
        if self.macro_state_labels is not None:
            unique_states, state_counts = np.unique(self.macro_state_labels, return_counts=True)
            plt.bar(unique_states, state_counts)
            plt.title('Macro-State Distribution')
            plt.xlabel('Macro-State ID')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio achieved by CFL"""
        return self.input_dim / self.n_macro_states
    
    def get_training_summary(self) -> dict:
        """Get a summary of the CFL training results"""
        if not self.is_trained:
            return {"error": "CFL not trained yet"}
            
        return {
            "input_dimension": self.input_dim,
            "macro_states": self.n_macro_states,
            "compression_ratio": self.get_compression_ratio(),
            "training_samples": len(self.transitions),
            "final_loss": self.training_losses['total'][-1] if self.training_losses['total'] else 0,
            "macro_state_distribution": dict(zip(*np.unique(self.macro_state_labels, return_counts=True)))
        }

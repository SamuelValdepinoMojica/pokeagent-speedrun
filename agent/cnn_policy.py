"""
Custom CNN policy for Pokemon Emerald DRL
Processes map tiles with CNN and combines with vector features
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict


class PokemonCNNExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Pokemon game state.
    
    Processes:
    - Map tiles (7x7x3) with CNN
    - Vector features (18) with linear layers
    - Combines both with fusion layer
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: Dictionary space with 'map' and 'vector' keys
            features_dim: Output dimension of the combined features
        """
        super().__init__(observation_space, features_dim)
        
        # Extract dimensions
        map_shape = observation_space['map'].shape  # (7, 7, 3)
        vector_dim = observation_space['vector'].shape[0]  # 18
        
        # === CNN for map processing ===
        # Input: (batch, 3, 7, 7) - channels first for PyTorch
        self.map_cnn = nn.Sequential(
            # Conv layer 1: 3 -> 32 channels
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Conv layer 2: 32 -> 64 channels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Flatten: 64 * 7 * 7 = 3136 features
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_map = torch.zeros(1, 3, 7, 7)
            cnn_output_size = self.map_cnn(sample_map).shape[1]
        
        # === MLP for vector features ===
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # === Fusion layer ===
        # Combine CNN output + MLP output
        combined_size = cnn_output_size + 128
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            observations: Dictionary with 'map' and 'vector' tensors
            
        Returns:
            Combined feature vector of size features_dim
        """
        # Extract observations
        map_obs = observations['map']  # (batch, 7, 7, 3)
        vector_obs = observations['vector']  # (batch, 18)
        
        # Transpose map from (batch, H, W, C) to (batch, C, H, W) for PyTorch CNN
        map_obs = map_obs.permute(0, 3, 1, 2)
        
        # Process map with CNN
        map_features = self.map_cnn(map_obs)
        
        # Process vector with MLP
        vector_features = self.vector_mlp(vector_obs)
        
        # Combine features
        combined = torch.cat([map_features, vector_features], dim=1)
        
        # Final fusion
        output = self.fusion(combined)
        
        return output


class PokemonCNNPolicy(ActorCriticPolicy):
    """
    Custom ActorCritic policy with CNN feature extractor.
    """
    
    def __init__(self, *args, **kwargs):
        # Set the custom feature extractor
        kwargs['features_extractor_class'] = PokemonCNNExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 256}
        
        # Call parent constructor
        super().__init__(*args, **kwargs)


# For easy import
__all__ = ['PokemonCNNExtractor', 'PokemonCNNPolicy']

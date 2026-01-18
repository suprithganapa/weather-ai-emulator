"""
Weather AI Emulator - Inference Module
Handles model loading and prediction
"""

import os
import torch
import numpy as np
from typing import Dict, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_lstm import CNNLSTMEmulator


class WeatherPredictor:
    """
    Weather prediction inference class.
    Loads trained models and makes predictions.
    """
    
    def __init__(self, checkpoint_dir='checkpoints'):
        """
        Initialize predictor.
        
        Args:
            checkpoint_dir: Directory containing trained model checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.models = {}
        self.stats = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration
        self.input_features = 5
        self.lookback = 6
        
        # Feature columns (must match training)
        self.feature_cols = ['rain', 'temp', 'wind', 'humidity', 'pressure']
        
        # Event columns (must match training)
        self.event_cols = [
            'cloudburst', 'thunderstorm', 'heatwave', 'coldwave',
            'cyclone_like', 'heavy_rain', 'high_wind', 'fog',
            'drought', 'humidity_extreme'
        ]
        
        print(f"WeatherPredictor initialized on device: {self.device}")
    
    def load_model(self, horizon: int):
        """
        Load model for a specific horizon.
        
        Args:
            horizon: Hours ahead (1, 3, 6, 12, or 24)
        """
        if horizon in self.models:
            return  # Already loaded
        
        # Construct paths
        model_path = os.path.join(self.checkpoint_dir, f'model_{horizon}h.pt')
        stats_path = os.path.join(self.checkpoint_dir, f'stats_{horizon}h.npy')
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}. "
                f"Please train the {horizon}h model first."
            )
        
        # Load normalization stats
        if os.path.exists(stats_path):
            self.stats[horizon] = np.load(stats_path, allow_pickle=True).item()
            print(f"Loaded normalization stats for {horizon}h model")
        else:
            print(f"Warning: Stats file not found: {stats_path}")
            # Use default stats (will be less accurate)
            self.stats[horizon] = {
                'feature_mean': np.zeros(self.input_features),
                'feature_std': np.ones(self.input_features)
            }
        
        # Create model
        model = CNNLSTMEmulator(
            input_features=self.input_features,
            cnn_channels=32,
            lstm_hidden=64,
            dropout=0.2
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()  # Set to evaluation mode
        
        self.models[horizon] = model
        print(f"✅ Loaded {horizon}h model from {model_path}")
    
    def normalize_input(self, weather_sequence: np.ndarray, horizon: int) -> np.ndarray:
        """
        Normalize input weather sequence.
        
        Args:
            weather_sequence: (lookback, features) raw weather data
            horizon: Horizon for which to normalize
        
        Returns:
            Normalized sequence
        """
        stats = self.stats[horizon]
        mean = stats['feature_mean']
        std = stats['feature_std']
        
        # Normalize
        normalized = (weather_sequence - mean) / (std + 1e-8)
        return normalized
    
    def predict(self, weather_sequence: List[Dict], horizon: int) -> Dict:
        """
        Make a prediction.
        
        Args:
            weather_sequence: List of recent weather observations
                Each dict should have keys: rain, temp, wind, humidity, pressure
            horizon: Hours ahead to predict (1, 3, 6, 12, or 24)
        
        Returns:
            Dict with keys: temperature, rainfall, wind, events
        """
        # Load model if not already loaded
        if horizon not in self.models:
            self.load_model(horizon)
        
        model = self.models[horizon]
        
        # Convert weather sequence to numpy array
        # Expected: last 6 hours of data
        if len(weather_sequence) < self.lookback:
            raise ValueError(
                f"Need at least {self.lookback} hours of data, "
                f"got {len(weather_sequence)}"
            )
        
        # Take last lookback hours
        recent_data = weather_sequence[-self.lookback:]
        
        # Extract features
        features = []
        for obs in recent_data:
            features.append([
                obs['rain'],
                obs['temp'],
                obs['wind'],
                obs['humidity'],
                obs['pressure']
            ])
        
        features = np.array(features, dtype=np.float32)  # (lookback, 5)
        
        # Normalize
        features_normalized = self.normalize_input(features, horizon)
        
        # Convert to tensor
        X = torch.FloatTensor(features_normalized).unsqueeze(0)  # (1, lookback, 5)
        X = X.to(self.device)
        
        # Predict
        with torch.no_grad():
            reg_out, cls_out = model(X)
        
        # Extract predictions
        reg_out = reg_out.cpu().numpy()[0]  # (3,)
        cls_out = cls_out.cpu().numpy()[0]  # (10,)
        
        # Parse regression outputs
        rainfall = float(reg_out[0])
        temperature = float(reg_out[1])
        wind = float(reg_out[2])
        
        # Parse classification outputs (event probabilities)
        events = {
            event: float(prob)
            for event, prob in zip(self.event_cols, cls_out)
        }
        
        return {
            'temperature': temperature,
            'rainfall': rainfall,
            'wind': wind,
            'events': events
        }


# Test function
if __name__ == '__main__':
    print("Testing WeatherPredictor...")
    
    predictor = WeatherPredictor(checkpoint_dir='../checkpoints')
    
    # Create fake weather sequence (last 6 hours)
    weather_sequence = [
        {'rain': 0.5, 'temp': 25.0, 'wind': 3.5, 'humidity': 70.0, 'pressure': 1013.0},
        {'rain': 0.3, 'temp': 25.5, 'wind': 3.2, 'humidity': 68.0, 'pressure': 1013.5},
        {'rain': 0.2, 'temp': 26.0, 'wind': 3.0, 'humidity': 65.0, 'pressure': 1014.0},
        {'rain': 0.1, 'temp': 26.5, 'wind': 2.8, 'humidity': 63.0, 'pressure': 1014.5},
        {'rain': 0.0, 'temp': 27.0, 'wind': 2.5, 'humidity': 60.0, 'pressure': 1015.0},
        {'rain': 0.0, 'temp': 27.5, 'wind': 2.3, 'humidity': 58.0, 'pressure': 1015.5},
    ]
    
    try:
        result = predictor.predict(weather_sequence, horizon=1)
        print("\n✅ Prediction successful!")
        print(f"Temperature: {result['temperature']:.2f}°C")
        print(f"Rainfall: {result['rainfall']:.2f}mm")
        print(f"Wind: {result['wind']:.2f}m/s")
        print("\nEvent probabilities:")
        for event, prob in result['events'].items():
            print(f"  {event}: {prob:.3f}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Please train the model first using train/train_cnn_lstm_1h.py")

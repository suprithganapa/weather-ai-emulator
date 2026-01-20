import torch
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_lstm import CNNLSTM

logger = logging.getLogger(__name__)

class WeatherPredictor:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cpu")
        self.horizons = [1, 3, 6, 12, 24]
        self.models = {}
        
        self.stats = {
            1: {'mean': np.array([0.79, 25.49, 3.42, 70.02, 1012.98]), 'std': np.array([1.59, 5.50, 1.45, 10.09, 3.01])},
            3: {'mean': np.array([0.79, 25.49, 3.42, 70.02, 1012.98]), 'std': np.array([1.59, 5.50, 1.45, 10.09, 3.01])},
            6: {'mean': np.array([0.79, 25.49, 3.42, 70.02, 1012.98]), 'std': np.array([1.59, 5.50, 1.45, 10.09, 3.01])},
            12: {'mean': np.array([0.79, 25.49, 3.42, 70.02, 1012.98]), 'std': np.array([1.59, 5.50, 1.45, 10.09, 3.01])},
            24: {'mean': np.array([0.79, 25.49, 3.42, 70.02, 1012.98]), 'std': np.array([1.59, 5.50, 1.45, 10.09, 3.01])},
        }
        
        # City-specific weather patterns (baseline values)
        self.city_baselines = {
            "Bangalore": {"temp": 24.5, "rain": 1.2, "wind": 2.8},
            "Mumbai": {"temp": 29.5, "rain": 2.5, "wind": 3.5},
            "Chennai": {"temp": 31.0, "rain": 1.8, "wind": 3.2},
            "Delhi": {"temp": 28.0, "rain": 0.8, "wind": 2.5},
            "Meghalaya": {"temp": 18.0, "rain": 5.5, "wind": 2.2},
            "Wayanad": {"temp": 22.0, "rain": 3.2, "wind": 2.0}
        }
        
        self.feature_names = ['rainfall', 'temperature', 'wind', 'humidity', 'pressure']
        self.event_names = ['cloudburst', 'thunderstorm', 'heatwave', 'coldwave', 'cyclone_like',
                           'heavy_rain', 'high_wind', 'fog', 'drought', 'humidity_extreme']
        self._load_models()
    
    def _load_models(self):
        for horizon in self.horizons:
            model_path = self.checkpoint_dir / f"model_{horizon}h.pt"
            
            try:
                model = CNNLSTM(input_size=5, hidden_size=64, num_classes=10)
                
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    
                    model.eval()
                    self.models[horizon] = model
                    logger.info(f"✅ Loaded {horizon}h model")
            except Exception as e:
                logger.error(f"❌ Failed loading {horizon}h: {e}")
    
    @property
    def models_loaded(self):
        return list(self.models.keys())
    
    def predict(self, city: str, horizon_hours: int, current_weather: Dict) -> Dict:
        if horizon_hours not in self.models:
            raise ValueError(f"No model for {horizon_hours}h")
        
        model = self.models[horizon_hours]
        stats = self.stats[horizon_hours]
        baseline = self.city_baselines.get(city, self.city_baselines["Bangalore"])
        
        # Use current weather + city baseline
        features = np.array([
            current_weather.get('rainfall', baseline['rain']),
            current_weather.get('temperature', baseline['temp']),
            current_weather.get('wind', baseline['wind']),
            current_weather.get('humidity', 70.0),
            current_weather.get('pressure', 1013.0)
        ])
        
        # Add horizon-based variations (weather changes over time)
        horizon_factor = horizon_hours / 24.0
        
        # Temperature variation
        temp_change = np.random.randn() * 2.5 * horizon_factor
        features[1] += temp_change
        
        # Rainfall variation (can increase or decrease)
        rain_change = np.random.randn() * 1.5 * horizon_factor
        features[0] = max(0, features[0] + rain_change)
        
        # Wind variation
        wind_change = np.random.randn() * 0.8 * horizon_factor
        features[2] = max(0, features[2] + wind_change)
        
        # Create lookback window
        window = np.zeros((6, 5))
        for i in range(6):
            age_factor = (6 - i) / 6
            variation = np.random.randn(5) * 0.5 * age_factor
            window[i] = features + variation
        
        # Normalize
        normalized = (window - stats['mean']) / (stats['std'] + 1e-8)
        
        # Predict
        x = torch.FloatTensor(normalized).unsqueeze(0)
        
        with torch.no_grad():
            reg_out, cls_out = model(x)
        
        # Get model predictions
        model_predictions = reg_out.numpy()[0] * stats['std'][:3] + stats['mean'][:3]
        event_probs = torch.sigmoid(cls_out).numpy()[0]
        
        # Blend model predictions with city baseline (70% model, 30% baseline)
        final_temp = 0.7 * model_predictions[1] + 0.3 * baseline['temp']
        final_rain = 0.7 * model_predictions[0] + 0.3 * baseline['rain']
        final_wind = 0.7 * model_predictions[2] + 0.3 * baseline['wind']
        
        # Add small random variation
        final_temp += np.random.randn() * 0.5
        final_rain += np.random.randn() * 0.3
        final_wind += np.random.randn() * 0.2
        
        return {
            "temperature": float(round(final_temp, 1)),
            "rainfall": float(round(max(0, final_rain), 1)),
            "wind": float(round(max(0, final_wind), 1)),
            "events": {event: float(prob) for event, prob in zip(self.event_names, event_probs)}
        }
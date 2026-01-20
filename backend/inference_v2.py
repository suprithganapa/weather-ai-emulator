import torch
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_lstm_improved import ImprovedCNNLSTM
logger = logging.getLogger(__name__)

class ImprovedWeatherPredictor:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cpu")
        self.horizons = [1, 3, 6, 12, 24]
        self.models = {}
        self.stats = {}
        self.event_names = ['cloudburst', 'thunderstorm', 'heatwave', 'coldwave', 'cyclone_like', 'heavy_rain', 'high_wind', 'fog', 'drought', 'humidity_extreme']
        self.city_baselines = {
            "Bangalore": {"temp": 24.0, "rain": 1.2, "wind": 2.5},
            "Mumbai": {"temp": 28.0, "rain": 2.5, "wind": 3.5},
            "Chennai": {"temp": 30.0, "rain": 1.5, "wind": 3.0},
            "Delhi": {"temp": 25.0, "rain": 0.8, "wind": 2.2},
            "Shillong": {"temp": 18.0, "rain": 8.0, "wind": 2.0},
            "Wayanad": {"temp": 22.0, "rain": 3.5, "wind": 1.8},
            "Jaipur": {"temp": 28.0, "rain": 0.3, "wind": 3.0},
            "Dharali": {"temp": 15.0, "rain": 4.0, "wind": 2.5},
            "Ladakh": {"temp": 8.0, "rain": 0.1, "wind": 4.0}
        }
        self._load_models()
    
    def _load_models(self):
        for horizon in self.horizons:
            try:
                model = ImprovedCNNLSTM(input_size=5, hidden_size=64, num_classes=10)
                model_path = self.checkpoint_dir / f"model_{horizon}h.pt"
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    model.eval()
                    self.models[horizon] = model
                    logger.info(f"✅ Loaded {horizon}h")
                stats_path = self.checkpoint_dir / f"stats_{horizon}h.npy"
                if stats_path.exists():
                    self.stats[horizon] = np.load(stats_path, allow_pickle=True).item()
            except Exception as e:
                logger.error(f"Failed {horizon}h: {e}")
    
    @property
    def models_loaded(self):
        return list(self.models.keys())
    
    def predict(self, city: str, horizon_hours: int, current_weather: Dict) -> Dict:
        if horizon_hours not in self.models:
            raise ValueError(f"No model for {horizon_hours}h")
        model = self.models[horizon_hours]
        stats = self.stats[horizon_hours]
        baseline = self.city_baselines.get(city, self.city_baselines["Bangalore"])
        features = np.array([current_weather.get('rainfall', baseline['rain']), current_weather.get('temperature', baseline['temp']), current_weather.get('wind', baseline['wind']), current_weather.get('humidity', 70.0), current_weather.get('pressure', 1013.0)])
        lookback = np.zeros((12, 5))
        for i in range(12):
            lookback[i] = features + np.random.randn(5) * np.array([0.3, 0.5, 0.2, 2.0, 1.0]) * (12-i)/12
        mean = torch.FloatTensor(stats['mean'])
        std = torch.FloatTensor(stats['std'])
        normalized = (torch.FloatTensor(lookback) - mean) / (std + 1e-8)
        with torch.no_grad():
            reg_out, cls_out = model(normalized.unsqueeze(0))
        predictions = reg_out.numpy()[0]
        event_probs = torch.sigmoid(cls_out).numpy()[0]
        return {"temperature": float(round(predictions[1], 1)), "rainfall": float(round(max(0, predictions[0]), 1)), "wind": float(round(max(0, predictions[2]), 1)), "events": {e: float(p) for e, p in zip(self.event_names, event_probs)}}
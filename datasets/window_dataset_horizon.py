"""
Weather Window Dataset with Variable Horizon
Supports multi-step ahead predictions: 1h, 3h, 6h, 12h, 24h
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class WeatherWindowDatasetHorizon(Dataset):
    """
    Multi-horizon sliding window dataset for weather prediction.
    
    Args:
        csv_path: Path to processed CSV file
        horizon_hours: Hours ahead to predict (1, 3, 6, 12, or 24)
        cities: List of cities to include (default: all)
        lookback: Number of past hours to use as input (default: 6)
    
    Returns:
        X: (lookback, 5) - Past weather features
        y_reg: (3,) - Regression targets: rain, temp, wind
        y_cls: (10,) - Classification targets: event probabilities
    """
    
    def __init__(self, csv_path, horizon_hours=1, cities=None, lookback=6):
        self.lookback = lookback
        self.horizon = horizon_hours
        
        # Load data
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Filter cities if specified
        if cities:
            df = df[df['city'].isin(cities)]
            print(f"Filtered to cities: {cities}")
            print(f"Remaining rows: {len(df)}")
        
        # Sort by city and time
        df = df.sort_values(['city', 'time']).reset_index(drop=True)
        
        # Define feature columns
        self.feature_cols = ['rain', 'temp', 'wind', 'humidity', 'pressure']
        
        # Define event columns
        self.event_cols = [
            'cloudburst', 'thunderstorm', 'heatwave', 'coldwave',
            'cyclone_like', 'heavy_rain', 'high_wind', 'fog',
            'drought', 'humidity_extreme'
        ]
        
        # Verify all columns exist
        missing_features = [col for col in self.feature_cols if col not in df.columns]
        missing_events = [col for col in self.event_cols if col not in df.columns]
        
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        if missing_events:
            raise ValueError(f"Missing event columns: {missing_events}")
        
        # Compute normalization statistics
        self.feature_mean = df[self.feature_cols].mean().values
        self.feature_std = df[self.feature_cols].std().values
        
        print(f"\nNormalization stats:")
        for i, col in enumerate(self.feature_cols):
            print(f"  {col}: mean={self.feature_mean[i]:.2f}, std={self.feature_std[i]:.2f}")
        
        # Create sliding windows with horizon offset
        self.X_windows = []
        self.y_reg = []
        self.y_cls = []
        
        print(f"\nCreating {horizon_hours}-hour ahead prediction windows...")
        for city in df['city'].unique():
            city_data = df[df['city'] == city].reset_index(drop=True)
            
            # Need at least lookback + horizon points for each city
            min_length = lookback + horizon_hours
            if len(city_data) < min_length:
                print(f"  Warning: {city} has only {len(city_data)} rows (need {min_length}), skipping")
                continue
            
            # Create windows
            city_windows = 0
            for i in range(lookback, len(city_data) - horizon_hours):
                # Input: past lookback hours at time i
                window = city_data.iloc[i-lookback:i][self.feature_cols].values
                
                # Target: future state at time i + horizon_hours
                target_row = city_data.iloc[i + horizon_hours]
                
                # Regression targets
                y_reg = target_row[['rain', 'temp', 'wind']].values.astype(np.float32)
                
                # Classification targets
                y_cls = target_row[self.event_cols].values.astype(np.float32)
                
                self.X_windows.append(window)
                self.y_reg.append(y_reg)
                self.y_cls.append(y_cls)
                city_windows += 1
            
            print(f"  {city}: {city_windows} windows")
        
        # Convert to numpy arrays
        self.X_windows = np.array(self.X_windows, dtype=np.float32)
        self.y_reg = np.array(self.y_reg, dtype=np.float32)
        self.y_cls = np.array(self.y_cls, dtype=np.float32)
        
        # Normalize input features
        self.X_windows = (self.X_windows - self.feature_mean) / (self.feature_std + 1e-8)
        
        print(f"\n✅ {horizon_hours}h-ahead dataset created successfully!")
        print(f"   Total samples: {len(self)}")
        print(f"   Input shape: {self.X_windows.shape}")
        print(f"   Regression output shape: {self.y_reg.shape}")
        print(f"   Classification output shape: {self.y_cls.shape}")
        print(f"   Event distribution: {self.y_cls.sum(axis=0).astype(int)}\n")
    
    def __len__(self):
        return len(self.X_windows)
    
    def __getitem__(self, idx):
        """Return a single sample."""
        X = torch.FloatTensor(self.X_windows[idx])
        y_reg = torch.FloatTensor(self.y_reg[idx])
        y_cls = torch.FloatTensor(self.y_cls[idx])
        return X, y_reg, y_cls
    
    def get_normalization_stats(self):
        """Return normalization statistics for inference."""
        return {
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'feature_cols': self.feature_cols,
            'horizon_hours': self.horizon
        }

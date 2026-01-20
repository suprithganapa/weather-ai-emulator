import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from models.cnn_lstm_v2 import ImprovedCNNLSTM

class WeatherDataset(Dataset):
    def __init__(self, data, cities, horizon_hours, lookback=12):
        self.data = data
        self.cities = cities
        self.horizon = horizon_hours
        self.lookback = lookback
        self.samples = self._create_samples()
    
    def _create_samples(self):
        samples = []
        
        for city in self.cities:
            city_data = self.data[self.data['city'] == city].reset_index(drop=True)
            
            for i in range(len(city_data) - self.lookback - self.horizon):
                # Input sequence
                input_seq = city_data.iloc[i:i+self.lookback][
                    ['rainfall', 'temperature', 'wind', 'humidity', 'pressure']
                ].values
                
                # Target (future values)
                target_idx = i + self.lookback + self.horizon - 1
                target_reg = city_data.iloc[target_idx][
                    ['rainfall', 'temperature', 'wind']
                ].values
                
                target_events = city_data.iloc[target_idx][[
                    'cloudburst', 'thunderstorm', 'heatwave', 'coldwave', 
                    'cyclone_like', 'heavy_rain', 'high_wind', 'fog', 
                    'drought', 'humidity_extreme'
                ]].values
                
                samples.append({
                    'input': input_seq,
                    'target_reg': target_reg,
                    'target_events': target_events
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['input']),
            torch.FloatTensor(sample['target_reg']),
            torch.FloatTensor(sample['target_events'])
        )

def train_model(horizon_hours, data, cities, epochs=100):
    """Train model for specific horizon"""
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL FOR {horizon_hours}h HORIZON")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create dataset
    dataset = WeatherDataset(data, cities, horizon_hours, lookback=12)
    print(f"Total samples: {len(dataset)}")
    
    # Train/val split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Model
    model = ImprovedCNNLSTM(input_size=5, hidden_size=128, num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss functions
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Compute normalization stats
    all_inputs = torch.stack([dataset[i][0] for i in range(min(1000, len(dataset)))])
    mean = all_inputs.mean(dim=(0, 1))
    std = all_inputs.std(dim=(0, 1))
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_reg_loss = 0
        train_cls_loss = 0
        
        for batch_idx, (inputs, targets_reg, targets_events) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets_reg = targets_reg.to(device)
            targets_events = targets_events.to(device)
            
            # Normalize
            inputs = (inputs - mean.to(device)) / (std.to(device) + 1e-8)
            
            optimizer.zero_grad()
            
            reg_out, cls_out = model(inputs)
            
            loss_reg = reg_criterion(reg_out, targets_reg)
            loss_cls = cls_criterion(cls_out, targets_events)
            loss = loss_reg + 0.5 * loss_cls
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_reg_loss += loss_reg.item()
            train_cls_loss += loss_cls.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_reg_loss = 0
        val_cls_loss = 0
        
        with torch.no_grad():
            for inputs, targets_reg, targets_events in val_loader:
                inputs = inputs.to(device)
                targets_reg = targets_reg.to(device)
                targets_events = targets_events.to(device)
                
                inputs = (inputs - mean.to(device)) / (std.to(device) + 1e-8)
                
                reg_out, cls_out = model(inputs)
                
                loss_reg = reg_criterion(reg_out, targets_reg)
                loss_cls = cls_criterion(cls_out, targets_events)
                loss = loss_reg + 0.5 * loss_cls
                
                val_loss += loss.item()
                val_reg_loss += loss_reg.item()
                val_cls_loss += loss_cls.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Reg: {train_reg_loss/len(train_loader):.4f}, Cls: {train_cls_loss/len(train_loader):.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (Reg: {val_reg_loss/len(val_loader):.4f}, Cls: {val_cls_loss/len(val_loader):.4f})")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs('checkpoints_v2', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, f'checkpoints_v2/model_{horizon_hours}h.pt')
            
            # Save stats
            np.save(f'checkpoints_v2/stats_{horizon_hours}h.npy', {
                'mean': mean.numpy(),
                'std': std.numpy()
            })
            
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining complete for {horizon_hours}h horizon!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return best_val_loss

def main():
    print("="*60)
    print("IMPROVED WEATHER MODEL TRAINING")
    print("="*60)
    
    # Load data
    data = pd.read_csv('data/weather_training_data.csv')
    cities = ['Bangalore', 'Mumbai', 'Chennai', 'Delhi', 'Shillong', 'Wayanad', 'Jaipur', 'Dharali']
    
    print(f"\nLoaded {len(data)} samples for {len(cities)} cities")
    print(f"Cities: {', '.join(cities)}")
    
    # Train all horizons
    horizons = [1, 3, 6, 12, 24]
    results = {}
    
    for horizon in horizons:
        val_loss = train_model(horizon, data, cities, epochs=100)
        results[horizon] = val_loss
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nResults:")
    for horizon, loss in results.items():
        print(f"  {horizon}h: Val Loss = {loss:.4f}")
    print("\nModels saved to: checkpoints_v2/")

if __name__ == "__main__":
    main()
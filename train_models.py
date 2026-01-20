import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from models.cnn_lstm_improved import ImprovedCNNLSTM

class WeatherDataset(Dataset):
    def __init__(self, data, cities, horizon_hours, lookback=12):
        self.samples = []
        for city in cities:
            city_data = data[data['city'] == city].reset_index(drop=True)
            for i in range(len(city_data) - lookback - horizon_hours):
                input_seq = city_data.iloc[i:i+lookback][['rainfall', 'temperature', 'wind', 'humidity', 'pressure']].values.astype(np.float32)
                target_idx = i + lookback + horizon_hours - 1
                target_reg = city_data.iloc[target_idx][['rainfall', 'temperature', 'wind']].values.astype(np.float32)
                target_events = city_data.iloc[target_idx][['cloudburst', 'thunderstorm', 'heatwave', 'coldwave', 'cyclone_like', 'heavy_rain', 'high_wind', 'fog', 'drought', 'humidity_extreme']].values.astype(np.float32)
                self.samples.append((torch.from_numpy(input_seq), torch.from_numpy(target_reg), torch.from_numpy(target_events)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def train_horizon(horizon_hours, data, cities, epochs=50):
    print(f'\nTRAINING {horizon_hours}h MODEL')
    dataset = WeatherDataset(data, cities, horizon_hours, lookback=12)
    print(f'Samples: {len(dataset):,}')
    train_size = int(0.85 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    model = ImprovedCNNLSTM(input_size=5, hidden_size=64, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    all_inputs = torch.stack([dataset[i][0] for i in range(min(500, len(dataset)))])
    mean = all_inputs.mean(dim=(0, 1))
    std = all_inputs.std(dim=(0, 1))
    best = 999
    patience = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets_reg, targets_events in train_loader:
            inputs = (inputs - mean) / (std + 1e-8)
            optimizer.zero_grad()
            reg_out, cls_out = model(inputs)
            loss = nn.MSELoss()(reg_out, targets_reg) + 0.5 * nn.BCEWithLogitsLoss()(cls_out, targets_events)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets_reg, targets_events in val_loader:
                    inputs = (inputs - mean) / (std + 1e-8)
                    reg_out, cls_out = model(inputs)
                    val_loss += nn.MSELoss()(reg_out, targets_reg).item()
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}/50 - Val: {val_loss:.4f}')
            if val_loss < best:
                best = val_loss
                patience = 0
                torch.save({'model_state_dict': model.state_dict()}, f'checkpoints/model_{horizon_hours}h.pt')
                np.save(f'checkpoints/stats_{horizon_hours}h.npy', {'mean': mean.numpy(), 'std': std.numpy()})
                print(f'  ✅ Saved!')
            else:
                patience += 1
                if patience >= 10:
                    print(f'  Early stop')
                    break
    print(f'Best val loss: {best:.4f}')
    return best

data = pd.read_csv('data/weather_training_data.csv')
cities = ['Bangalore', 'Mumbai', 'Chennai', 'Delhi', 'Shillong', 'Wayanad', 'Jaipur', 'Dharali', 'Ladakh']
print('='*70)
print('TRAINING ALL 9 CITIES, 5 HORIZONS - 50 EPOCHS EACH')
print('='*70)
results = {}
for h in [1, 3, 6, 12, 24]:
    results[h] = train_horizon(h, data, cities, epochs=50)
print('\n' + '='*70)
print('TRAINING COMPLETE!')
print('='*70)
for h, loss in results.items():
    print(f'{h:2d}h: Val Loss = {loss:.4f}')
print('='*70)
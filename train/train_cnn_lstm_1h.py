"""
Train CNN-LSTM model for 1-hour ahead weather prediction
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.window_dataset_horizon import WeatherWindowDatasetHorizon
from models.cnn_lstm import CNNLSTMEmulator

# Configuration
CITIES = ['Bangalore', 'Mumbai', 'Meghalaya', 'Wayanad', 'Chennai', 'Delhi']
CSV_PATH = 'data/processed/nasa_power_labeled_v2.csv'
CHECKPOINT_DIR = '../checkpoints'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model_1h.pt')
STATS_PATH = os.path.join(CHECKPOINT_DIR, 'stats_1h.npy')

HORIZON_HOURS = 1
LOOKBACK = 6
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, loader, reg_loss_fn, cls_loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_reg_loss = 0
    total_cls_loss = 0
    
    for X, y_reg, y_cls in tqdm(loader, desc='Training', leave=False):
        X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_reg, pred_cls = model(X)
        
        # Compute losses
        loss_reg = reg_loss_fn(pred_reg, y_reg)
        loss_cls = cls_loss_fn(pred_cls, y_cls)
        loss = loss_reg + loss_cls
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_reg_loss += loss_reg.item()
        total_cls_loss += loss_cls.item()
    
    return total_loss / len(loader), total_reg_loss / len(loader), total_cls_loss / len(loader)


def validate(model, loader, reg_loss_fn, cls_loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_reg_loss = 0
    total_cls_loss = 0
    
    with torch.no_grad():
        for X, y_reg, y_cls in tqdm(loader, desc='Validating', leave=False):
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            
            # Forward pass
            pred_reg, pred_cls = model(X)
            
            # Compute losses
            loss_reg = reg_loss_fn(pred_reg, y_reg)
            loss_cls = cls_loss_fn(pred_cls, y_cls)
            loss = loss_reg + loss_cls
            
            total_loss += loss.item()
            total_reg_loss += loss_reg.item()
            total_cls_loss += loss_cls.item()
    
    return total_loss / len(loader), total_reg_loss / len(loader), total_cls_loss / len(loader)


def main():
    print("="*60)
    print(f"Training CNN-LSTM Model: {HORIZON_HOURS}-hour ahead prediction")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Cities: {CITIES}")
    print(f"Lookback: {LOOKBACK} hours")
    print(f"Horizon: {HORIZON_HOURS} hour(s)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*60 + "\n")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        print("Please ensure your dataset is in the correct location.")
        return
    
    # Load dataset
    try:
        dataset = WeatherWindowDatasetHorizon(
            CSV_PATH, 
            horizon_hours=HORIZON_HOURS,
            cities=CITIES,
            lookback=LOOKBACK
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("❌ Error: Dataset is empty!")
        return
    
    # Save normalization stats
    stats = dataset.get_normalization_stats()
    np.save(STATS_PATH, stats)
    print(f"✅ Saved normalization stats to {STATS_PATH}\n")
    
    # Split dataset: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_ds)} samples ({100*len(train_ds)/len(dataset):.1f}%)")
    print(f"  Val:   {len(val_ds)} samples ({100*len(val_ds)/len(dataset):.1f}%)")
    print(f"  Test:  {len(test_ds)} samples ({100*len(test_ds)/len(dataset):.1f}%)\n")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    model = CNNLSTMEmulator(
        input_features=5,
        cnn_channels=32,
        lstm_hidden=64,
        dropout=0.2
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss functions
    reg_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.BCELoss()
    
    # Optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Starting training...\n")
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_reg_loss, train_cls_loss = train_one_epoch(
            model, train_loader, reg_loss_fn, cls_loss_fn, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_reg_loss, val_cls_loss = validate(
            model, val_loader, reg_loss_fn, cls_loss_fn, DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} (Reg: {train_reg_loss:.4f}, Cls: {train_cls_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Reg: {val_reg_loss:.4f}, Cls: {val_cls_loss:.4f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, CHECKPOINT_PATH)
            print(f"  ✅ Best model saved (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹ Early stopping triggered (no improvement for {patience} epochs)")
            break
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_reg_loss, test_cls_loss = validate(
        model, test_loader, reg_loss_fn, cls_loss_fn, DEVICE
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"  Regression Loss (MSE): {test_reg_loss:.4f}")
    print(f"  Classification Loss (BCE): {test_cls_loss:.4f}")
    
    # Compute regression metrics (MAE)
    model.eval()
    mae_rain, mae_temp, mae_wind = 0, 0, 0
    num_batches = 0
    
    with torch.no_grad():
        for X, y_reg, y_cls in test_loader:
            X, y_reg = X.to(DEVICE), y_reg.to(DEVICE)
            pred_reg, _ = model(X)
            
            mae = torch.abs(pred_reg - y_reg).mean(dim=0)
            mae_rain += mae[0].item()
            mae_temp += mae[1].item()
            mae_wind += mae[2].item()
            num_batches += 1
    
    mae_rain /= num_batches
    mae_temp /= num_batches
    mae_wind /= num_batches
    
    print(f"\nRegression MAE:")
    print(f"  Rain: {mae_rain:.4f} mm")
    print(f"  Temperature: {mae_temp:.4f} °C")
    print(f"  Wind: {mae_wind:.4f} m/s")
    
    print("\n" + "="*60)
    print(f"✅ Training complete! Model saved to {CHECKPOINT_PATH}")
    print("="*60)


if __name__ == '__main__':
    main()

"""
CNN-LSTM Weather Emulator Model
Combines CNN for feature extraction with LSTM for temporal modeling
"""

import torch
import torch.nn as nn


class CNNLSTMEmulator(nn.Module):
    """
    Hybrid CNN-LSTM architecture for weather prediction.
    
    Architecture:
        1. CNN: Extract spatial/feature patterns
        2. LSTM: Model temporal dependencies
        3. Regression head: Predict rain, temp, wind
        4. Classification head: Predict 10 extreme event probabilities
    
    Args:
        input_features: Number of input features (default: 5)
        cnn_channels: Number of CNN output channels (default: 32)
        lstm_hidden: LSTM hidden dimension (default: 64)
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(self, input_features=5, cnn_channels=32, lstm_hidden=64, dropout=0.2):
        super(CNNLSTMEmulator, self).__init__()
        
        self.input_features = input_features
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        
        # CNN for feature extraction
        # Input: (batch, lookback, features)
        # Conv1d expects (batch, features, lookback)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.Dropout(dropout),
        )
        
        # LSTM for temporal modeling
        # Input: (batch, lookback, features)
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        
        # Regression head: predict rain, temp, wind (3 values)
        self.reg_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)  # rain, temp, wind
        )
        
        # Classification head: predict 10 extreme event probabilities
        self.cls_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 10),  # 10 event types
            nn.Sigmoid()  # probabilities [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, lookback, features) - Input weather sequence
        
        Returns:
            reg_out: (batch, 3) - Regression predictions
            cls_out: (batch, 10) - Event probabilities
        """
        batch_size, lookback, features = x.shape
        
        # CNN: (batch, lookback, features) -> (batch, features, lookback)
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn(x_cnn)
        
        # Back to (batch, lookback, cnn_channels*2)
        x_cnn = x_cnn.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_cnn)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        
        # Regression output
        reg_out = self.reg_head(last_hidden)  # (batch, 3)
        
        # Classification output
        cls_out = self.cls_head(last_hidden)  # (batch, 10)
        
        return reg_out, cls_out


# Test function
if __name__ == '__main__':
    print("Testing CNNLSTMEmulator...")
    
    # Create model
    model = CNNLSTMEmulator()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    lookback = 6
    features = 5
    
    x = torch.randn(batch_size, lookback, features)
    reg_out, cls_out = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Regression output shape: {reg_out.shape} (expected: {batch_size}, 3)")
    print(f"Classification output shape: {cls_out.shape} (expected: {batch_size}, 10)")
    print(f"\nSample regression output: {reg_out[0].detach().numpy()}")
    print(f"Sample classification output: {cls_out[0].detach().numpy()}")
    print("\n✅ Model test passed!")

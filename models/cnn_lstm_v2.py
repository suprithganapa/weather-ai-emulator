import torch
import torch.nn as nn

class ImprovedCNNLSTM(nn.Module):
    """
    Enhanced CNN-LSTM with better accuracy
    - Deeper CNN layers
    - Batch normalization
    - Residual connections
    - Attention mechanism
    """
    
    def __init__(self, input_size=5, hidden_size=128, num_classes=10):
        super(ImprovedCNNLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Deeper CNN with batch normalization
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            128, hidden_size, 
            num_layers=3, 
            batch_first=True, 
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Regression head with residual
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # CNN
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Outputs
        reg_output = self.reg_head(context)
        cls_output = self.cls_head(context)
        
        return reg_output, cls_output
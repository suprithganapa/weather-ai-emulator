import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    """CNN-LSTM model for weather prediction"""
    
    def __init__(self, input_size=5, hidden_size=64, num_classes=10):
        super(CNNLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # LSTM
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        
        # Output heads
        self.fc_reg = nn.Linear(hidden_size, 3)
        self.fc_cls = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # CNN
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.dropout(x)
        x = self.relu2(self.conv2(x))
        x = self.dropout(x)
        
        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Outputs
        reg_output = self.fc_reg(last_hidden)
        cls_output = self.fc_cls(last_hidden)
        
        return reg_output, cls_output
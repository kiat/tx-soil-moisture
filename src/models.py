import torch
import torch.nn as nn

#PyTorch Baseline Model

class Baseline(nn.Module):
    """
    A baseline model that predicts the last known value of the first feature.
    Converted to a non-trainable PyTorch module.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input x is expected to be a tensor of shape (batch, sequence_length, features)
        # Slicing is the same, but now it's a PyTorch tensor operation.
        return x[:, -1, 0].unsqueeze(-1) # Return shape (batch, 1) for consistency

class MovingAverageBaseline(nn.Module):
    """
    A baseline that predicts the average of the last N values of the first feature.
    Converted to a non-trainable PyTorch module.
    """
    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        # Input x is expected to be a tensor of shape (batch, sequence_length, features)
        N = self.window_size
        if N > x.shape[1]:
            raise ValueError(f"Window size {N} is larger than input sequence length {x.shape[1]}")
        
        # Slicing and mean operations are now PyTorch native.
        return torch.mean(x[:, -N:, 0], dim=1).unsqueeze(-1) # Return shape (batch, 1)

# --- PyTorch Models ---

class AR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # PyTorch Conv1D/Pool1D expect (batch, features, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.proj_inp = nn.Linear(input_dim, 64)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=16, hidden_size=8, batch_first=True)
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x  = self.proj_inp(x)  # Project input to 64 dimensions
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)  # x shape: (batch, seq_len, 8)
        # Using the output of the very last time step for prediction
        x = x[:, -1, :]      
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.proj_inp = nn.Linear(input_dim, 64)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=16, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=8, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(16, 8) # 2 * hidden_size (8) because of bidirectionality
        self.fc2 = nn.Linear(8, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x  = self.proj_inp(x)  # Project input to 64 dimensions
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        _, (hn, _) = self.lstm3(x)
        # Correctly concatenate final forward (hn[-2]) and backward (hn[-1]) hidden states
        x = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.proj_inp = nn.Linear(input_dim, 64)
        self.rnn = nn.RNN(input_size=64, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x  = self.proj_inp(x)  # Project input to 64 dimensions
        _, hn = self.rnn(x) # Get the last hidden state
        x = hn.squeeze(0)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNModel(nn.Module):
    # Now robust to changes in sequence length
    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        # Add padding=1 to handle small sequence lengths (kernel_size=3 requires at least 3 timesteps)
        self.proj_inp = nn.Linear(input_dim, 64)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1) # Added pooling for fixed-size output
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # PyTorch Conv1D expects (batch, features, sequence_length)
        x  = self.proj_inp(x)  # Project input to 64 dimensions
        x = x.permute(0, 2, 1)
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.proj_inp = nn.Linear(input_dim, 64)
        self.lstm1 = nn.LSTM(64, 32, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x  = self.proj_inp(x)  # Project input to 64 dimensions
        lstm_out, _ = self.lstm1(x)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        _, (hn, _) = self.lstm2(attn_output)
        x = hn.squeeze(0)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AttentionOnly(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        # Ensure num_heads divides input_dim evenly
        num_heads = min(4, input_dim) if input_dim < 4 else 4
        while input_dim % num_heads != 0:
            num_heads -= 1
        if num_heads == 0:
            num_heads = 1
            
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(attn_output + x) # Added residual connection, common practice
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        
        # Ensure num_heads divides input_dim evenly
        if input_dim % num_heads != 0:
            # Find largest valid num_heads
            num_heads = min(num_heads, input_dim)
            while input_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            if num_heads == 0:
                num_heads = 1
                
        # Using a standard Transformer Encoder Layer is cleaner
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=num_heads, 
            dim_feedforward=64,
            batch_first=True
        )
        self.proj_inp = nn.Linear(input_dim, 64)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x  = self.proj_inp(x)  # Project input to 64 dimensions
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiHeadLSTM(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        
        # For MultiHeadAttention on LSTM output (embed_dim=32), ensure num_heads divides 32
        if 32 % num_heads != 0:
            num_heads = min(num_heads, 32)
            while 32 % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            if num_heads == 0:
                num_heads = 1
                
        self.proj_inp = nn.Linear(input_dim, 64)
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x  = self.proj_inp(x)  # Project input to 64 dimensions
        lstm_out, _ = self.lstm(x)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # Pool the features across the sequence length from the attention output
        x = attn_output.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
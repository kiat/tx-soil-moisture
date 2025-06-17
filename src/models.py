import torch
import torch.nn as nn
import numpy as np

# Baseline models (no changes needed, as they don't use a deep learning framework)
class Baseline:
    def fit(self, X, y, *args, **kwargs):
        return self

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return X[:, -1, 0]

class MovingAverageBaseline:
    def __init__(self, window_size=3):
        self.window_size = window_size

    def fit(self, X, y, *args, **kwargs):
        return self

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        N = self.window_size
        if N > X.shape[1]:
            raise ValueError(f"Window size {N} is larger than input sequence length {X.shape[1]}")
        return X[:, -N:, 0].mean(axis=1)

# PyTorch Models
class AR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
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
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=16, hidden_size=8, batch_first=True)
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, (hn, _) = self.lstm3(x)
        x = hn.squeeze(0) # Get the last hidden state
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=16, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=8, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(16, 8) # Bidirectional output is 2 * hidden_size
        self.fc2 = nn.Linear(8, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, (hn, _) = self.lstm3(x)
        # Concatenate final forward and backward hidden states
        x = torch.cat((hn[0,:,:], hn[1,:,:]), dim=1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, hn = self.rnn(x) # Get the last hidden state
        x = hn.squeeze(0)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, seq_len, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3)
        # Calculate the flattened size after convolution
        flat_size = 32 * (seq_len - 3 + 1)
        self.fc1 = nn.Linear(flat_size, 8)
        self.fc2 = nn.Linear(8, 1)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # PyTorch Conv1D expects (batch, features, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.tanh(self.conv1(x))
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 32, batch_first=True, return_sequences=True)
        # Using PyTorch's MultiheadAttention as a self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        # For self-attention, query, key, and value are the same
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        _, (hn, _) = self.lstm2(attn_output)
        x = hn.squeeze(0)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AttentionOnly(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, key_dim=16, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(attn_output)
        # Pool across the sequence dimension
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, key_dim=16):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, key_dim=key_dim, batch_first=True)
        self.add_norm1 = nn.LayerNorm(input_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.add_norm2 = nn.LayerNorm(input_dim)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Attention Block
        attn_output, _ = self.attention(x, x, x)
        x = self.add_norm1(x + attn_output)
        
        # Feed-forward Block
        ffn_output = self.ffn(x)
        x = self.add_norm2(x + ffn_output)
        
        # Pooling and Final Layers
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiHeadLSTM(nn.Module):
    def __init__(self, input_dim, num_heads=4, key_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, key_dim=key_dim, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = attn_output.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
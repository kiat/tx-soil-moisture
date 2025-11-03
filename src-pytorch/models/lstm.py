"""
LSTM-based models for time-series prediction.

This module provides:
- LSTMModel: Standard LSTM with projection layer
- BiLSTMModel: Bidirectional LSTM with layer normalization
- MultiHeadLSTM: LSTM with multi-head attention

Purpose:
    LSTM models are well-suited for sequence modeling tasks because they
    can learn long-term dependencies. These implementations include modern
    improvements like input projection, layer normalization, and attention.

Usage:
    >>> model = LSTMModel(input_dim=10, hidden_size=64, num_layers=3)
    >>> predictions = model(input_tensor)  # (batch, seq_len, features)
"""

import torch
import torch.nn as nn
from .base import BaseModel


class LSTMModel(BaseModel):
    """
    Standard LSTM model with input projection and multi-layer architecture.

    Features:
    - Projects input to a fixed hidden size before LSTM processing
    - Stacked LSTM layers with dropout for regularization
    - MLP decoder for final prediction

    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output features (default: 1)
        num_layers (int): Number of LSTM layers (default: 3)
        hidden_size (int): Hidden state dimension (default: 64)
        dropout (float): Dropout probability between layers (default: 0.2)

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, output_dim)
    """

    def __init__(
        self, input_dim, output_dim=1, num_layers=3, hidden_size=64, dropout=0.2
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Project input to hidden dimension
        self.proj_inp = nn.Linear(input_dim, hidden_size)

        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Dropout only between layers
        )

        # MLP decoder
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, output_dim)
        """
        x = self.proj_inp(x)  # (batch, seq_len, hidden_size)
        x, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Use the output of the last time step
        x = x[:, -1, :]  # (batch, hidden_size)
        x = self.fc(x)  # (batch, output_dim)
        return x


class BiLSTMModel(BaseModel):
    """
    Bidirectional LSTM model with layer normalization and dropout.

    Features:
    - Processes sequences in both forward and backward directions
    - Layer normalization after concatenating bidirectional outputs
    - Dropout for regularization
    - Uses final hidden states (not last output)

    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output features (default: 1)
        num_layers (int): Number of BiLSTM layers (default: 2)
        hidden_size (int): Hidden state dimension per direction (default: 64)
        dropout (float): Dropout probability (default: 0.2)

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, output_dim)

    Note:
        Output dimension from BiLSTM is 2 * hidden_size due to concatenation
        of forward and backward hidden states.
    """

    def __init__(
        self, input_dim, output_dim=1, num_layers=2, hidden_size=64, dropout=0.2
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Project input to hidden dimension
        self.proj_inp = nn.Linear(input_dim, hidden_size)

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Normalization and regularization
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)

        # MLP decoder
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        """
        Forward pass through the BiLSTM model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, output_dim)
        """
        x = self.proj_inp(x)  # (batch, seq_len, hidden_size)
        x, (hn, _) = self.bilstm(x)  # hn: (num_layers * 2, batch, hidden_size)

        # Concatenate final forward and backward hidden states
        forward = hn[-2, :, :]  # (batch, hidden_size)
        backward = hn[-1, :, :]  # (batch, hidden_size)
        x = torch.cat((forward, backward), dim=1)  # (batch, 2 * hidden_size)

        # Apply normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Decode to predictions
        x = self.fc(x)  # (batch, output_dim)
        return x


class MultiHeadLSTM(BaseModel):
    """
    LSTM model with multi-head self-attention mechanism.

    Features:
    - LSTM for sequential processing
    - Multi-head attention to focus on important parts of the sequence
    - Adaptive pooling for flexible sequence lengths

    Args:
        input_dim (int): Number of input features
        num_heads (int): Number of attention heads (default: 4)
                        Will be adjusted to divide embed_dim evenly

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, 1)

    Note:
        The attention mechanism operates on LSTM outputs with embed_dim=32.
        num_heads will be automatically adjusted to divide 32 evenly.
    """

    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Adjust num_heads to divide 32 (LSTM output dim) evenly
        if 32 % num_heads != 0:
            num_heads = min(num_heads, 32)
            while 32 % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            if num_heads == 0:
                num_heads = 1

        # Input projection
        self.proj_inp = nn.Linear(input_dim, 64)

        # LSTM layer
        self.lstm = nn.LSTM(64, 32, batch_first=True)

        # Multi-head attention on LSTM outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=num_heads, batch_first=True
        )

        # Pooling and decoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the MultiHeadLSTM model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, 1)
        """
        x = self.proj_inp(x)  # (batch, seq_len, 64)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 32)

        # Apply self-attention
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # (batch, seq_len, 32)

        # Pool across sequence dimension
        x = attn_output.permute(0, 2, 1)  # (batch, 32, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, 32)

        # Decode to prediction
        x = self.relu(self.fc1(x))  # (batch, 8)
        x = self.fc2(x)  # (batch, 1)
        return x

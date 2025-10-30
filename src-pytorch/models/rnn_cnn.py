"""
RNN and CNN models for time-series prediction.

This module provides:
- RNNModel: Simple RNN with projection
- CNNModel: 1D CNN with adaptive pooling

Purpose:
    These models provide alternative architectures to LSTMs:
    - RNN: Simpler and faster than LSTM, good for shorter sequences
    - CNN: Captures local patterns, parallel processing

Usage:
    >>> rnn_model = RNNModel(input_dim=10)
    >>> cnn_model = CNNModel(input_dim=10)
"""

import torch
import torch.nn as nn
from .base import BaseModel


class RNNModel(BaseModel):
    """
    Simple RNN model with input projection and MLP decoder.

    Features:
    - Projects input to fixed dimension
    - Single-layer vanilla RNN
    - Uses final hidden state for prediction

    Args:
        input_dim (int): Number of input features

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, 1)

    Note:
        RNNs are simpler than LSTMs but may struggle with long sequences
        due to vanishing gradient problems. Good for short-term dependencies.
    """

    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Input projection
        self.proj_inp = nn.Linear(input_dim, 64)

        # Simple RNN layer
        self.rnn = nn.RNN(input_size=64, hidden_size=32, batch_first=True)

        # MLP decoder
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the RNN model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, 1)
        """
        x = self.proj_inp(x)  # (batch, seq_len, 64)
        _, hn = self.rnn(x)  # hn: (1, batch, 32)
        x = hn.squeeze(0)  # (batch, 32)
        x = self.relu(self.fc1(x))  # (batch, 8)
        x = self.fc2(x)  # (batch, 1)
        return x


class CNNModel(BaseModel):
    """
    1D Convolutional Neural Network for time-series prediction.

    Features:
    - Projects input to higher dimension
    - 1D convolution to capture local temporal patterns
    - Adaptive pooling handles variable sequence lengths
    - Tanh activation for bounded outputs

    Args:
        input_dim (int): Number of input features

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, 1)

    Note:
        CNNs can process sequences in parallel (unlike RNNs) and are effective
        at capturing local patterns. The adaptive pooling makes this model
        robust to different sequence lengths.
    """

    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Input projection to higher dimension
        self.proj_inp = nn.Linear(input_dim, 64)

        # 1D convolution with padding to preserve sequence length
        self.conv1 = nn.Conv1d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1
        )

        # Adaptive pooling aggregates across entire sequence
        self.pool = nn.AdaptiveAvgPool1d(1)

        # MLP decoder
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Forward pass through the CNN model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, 1)
        """
        # Project input features
        x = self.proj_inp(x)  # (batch, seq_len, 64)

        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, 64, seq_len)
        x = self.tanh(self.conv1(x))  # (batch, 32, seq_len)

        # Pool across sequence dimension
        x = self.pool(x)  # (batch, 32, 1)
        x = self.flatten(x)  # (batch, 32)

        # Decode to prediction
        x = self.tanh(self.fc1(x))  # (batch, 8)
        x = self.fc2(x)  # (batch, 1)
        return x

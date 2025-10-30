"""
Transformer and other advanced models for time-series prediction.

This module provides:
- AR: Autoregressive model with adaptive pooling
- TransformerModel: Transformer encoder for time-series
- (Future models can be added here)

Purpose:
    These models represent more advanced architectures:
    - AR: Simple autoregressive baseline with learned features
    - Transformer: State-of-the-art architecture using self-attention

Usage:
    >>> ar_model = AR(input_dim=10)
    >>> transformer_model = TransformerModel(input_dim=10, num_heads=4)
"""

import torch
import torch.nn as nn
from .base import BaseModel


class AR(BaseModel):
    """
    Autoregressive model with adaptive pooling.

    Features:
    - Adaptive average pooling to compress sequence to single timestep
    - Two-layer MLP for prediction
    - Simple but effective baseline

    Args:
        input_dim (int): Number of input features

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, 1)

    Note:
        Despite its simplicity, this model can be surprisingly effective
        for stationary time series with strong autoregressive properties.
    """

    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Adaptive pooling compresses sequence dimension to 1
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Two-layer MLP
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the AR model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, 1)
        """
        # Conv1D/Pool1D expect (batch, features, sequence_length)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.pool(x)  # (batch, input_dim, 1)
        x = x.squeeze(-1)  # (batch, input_dim)
        x = self.relu(self.fc1(x))  # (batch, 64)
        x = self.fc2(x)  # (batch, 1)
        return x


class TransformerModel(BaseModel):
    """
    Transformer encoder model for time-series prediction.

    Features:
    - Transformer encoder with self-attention
    - Input projection to fixed dimension
    - Adaptive pooling for final aggregation
    - State-of-the-art architecture for sequence modeling

    Args:
        input_dim (int): Number of input features
        num_heads (int): Number of attention heads (default: 4)
                        Will be adjusted to divide d_model evenly

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, 1)

    Note:
        Transformers process sequences in parallel and can capture both
        short and long-range dependencies. They typically require more
        data than RNNs to train effectively.
    """

    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        # Model dimension for transformer
        d_model = 64

        # Ensure num_heads divides d_model evenly
        if d_model % num_heads != 0:
            # Find largest valid num_heads
            num_heads = min(num_heads, d_model)
            while d_model % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            if num_heads == 0:
                num_heads = 1

        # Input projection to transformer dimension
        self.proj_inp = nn.Linear(input_dim, d_model)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model,  # Feedforward dimension
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Pooling and decoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the Transformer model.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            Predictions (batch, 1)
        """
        # Project input to transformer dimension
        x = self.proj_inp(x)  # (batch, seq_len, d_model)

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Pool across sequence dimension
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)

        # Decode to prediction
        x = self.relu(self.fc1(x))  # (batch, 32)
        x = self.fc2(x)  # (batch, 1)
        return x

"""
Baseline models for time-series prediction.

This module provides:
- Baseline: Last-value predictor (persistence model)
- MovingAverageBaseline: Moving average predictor

Purpose:
    These non-trainable models serve as simple baselines to compare
    against more complex neural network models. They establish a
    performance floor that any ML model should exceed.

Usage:
    >>> model = Baseline(label_width=1)
    >>> predictions = model(input_tensor)  # (batch, seq_len, features)
"""

import torch
import torch.nn as nn
from .base import BaseModel


class Baseline(BaseModel):
    """
    A baseline model that predicts the last known value of the first feature.

    This is also known as a "persistence model" or "naive forecast".
    It assumes that the next value will be the same as the most recent value.

    Args:
        label_width (int): Number of future timesteps to predict (default: 1)

    Input shape:
        (batch_size, sequence_length, num_features)

    Output shape:
        (batch_size, label_width) if label_width > 1
        (batch_size, 1) if label_width == 1
    """

    def __init__(self, label_width=1):
        super().__init__()
        self.label_width = label_width

    def forward(self, x):
        """
        Predict the last known value for future timesteps.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)

        Returns:
            Predicted values (batch, label_width)
        """
        # Extract the last value of the first feature
        last_value = x[:, -1, 0]

        if self.label_width == 1:
            return last_value.unsqueeze(-1)  # (batch, 1)
        else:
            # Repeat the last value for all future timesteps
            return last_value.unsqueeze(-1).repeat(1, self.label_width)


class MovingAverageBaseline(BaseModel):
    """
    A baseline that predicts the average of the last N values of the first feature.

    This smooths out short-term fluctuations and can outperform simple persistence
    when the signal is noisy.

    Args:
        window_size (int): Number of recent values to average (default: 3)
        label_width (int): Number of future timesteps to predict (default: 1)

    Input shape:
        (batch_size, sequence_length, num_features)
        where sequence_length >= window_size

    Output shape:
        (batch_size, 1)
    """

    def __init__(self, window_size=3, label_width=1):
        super().__init__()
        self.window_size = window_size
        self.label_width = label_width

    def forward(self, x):
        """
        Predict the moving average of recent values.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)

        Returns:
            Predicted values (batch, 1)

        Raises:
            ValueError: If window_size > input sequence length
        """
        N = self.window_size
        if N > x.shape[1]:
            raise ValueError(
                f"Window size {N} is larger than input sequence length {x.shape[1]}"
            )

        # Compute mean of the last N values of the first feature
        return torch.mean(x[:, -N:, 0], dim=1).unsqueeze(-1)  # (batch, 1)

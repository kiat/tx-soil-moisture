"""1D-CNN architectures for soil-moisture regression.

Three models are provided:

* ``SmallCNN``        -- 1 conv layer (8 filters), ~73 params at F=1.
* ``LargerCNN``       -- 2 conv layers (16->32 filters), ~2.2k params at F=1.
  This is the main model used throughout the multi-station experiments.
* ``ConfigurableCNN`` -- same family with tunable kernel size, depth, width,
  and dropout, used for the hyperparameter sweep.

Input is ``(batch, window, features)`` and is transposed internally to the
``(batch, channels, time)`` layout expected by ``nn.Conv1d``.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """One convolutional layer, average pooling, small MLP head."""

    def __init__(self, num_features: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(num_features, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)))


class LargerCNN(nn.Module):
    """Two convolutional layers (16 -> 32), average pooling, MLP head."""

    def __init__(self, num_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)))


class ConfigurableCNN(nn.Module):
    """CNN with tunable kernel size, depth, base filter count, and dropout."""

    def __init__(self, num_features: int, kernel_size: int = 3, n_layers: int = 2,
                 base_filters: int = 16, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        layers, cin, cout = [], num_features, base_filters
        for _ in range(n_layers):
            layers += [nn.Conv1d(cin, cout, kernel_size, padding=pad), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            cin, cout = cout, cout * 2
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(cin, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)))


def build_model(num_features: int, arch: str = "larger", **kw) -> nn.Module:
    """Factory: ``arch`` in {small, larger, configurable}."""
    if arch == "small":
        return SmallCNN(num_features)
    if arch == "larger":
        return LargerCNN(num_features)
    if arch == "configurable":
        return ConfigurableCNN(
            num_features,
            kernel_size=kw.get("kernel_size", 3),
            n_layers=kw.get("n_layers", 2),
            base_filters=kw.get("base_filters", 16),
            dropout=kw.get("dropout", 0.0),
        )
    raise ValueError(f"unknown arch {arch!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

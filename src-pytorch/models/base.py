"""
Base model class for all PyTorch models in this package.

This module provides:
- BaseModel: A parent class with common utilities for save/load operations
- Shared constants and helper functions used across multiple models

Purpose:
    Standardize model lifecycle operations (save, load, device management)
    and provide a consistent interface for all model classes.
"""

import torch
import torch.nn as nn
import os


class BaseModel(nn.Module):
    """
    Base class for all time-series prediction models.

    Provides standardized save/load functionality and device management.
    All custom models should inherit from this class to ensure consistency.
    """

    def save_checkpoint(self, path, optimizer=None, epoch=None, metadata=None):
        """
        Save model checkpoint with optional training state.

        Args:
            path (str): File path to save checkpoint
            optimizer: Optional optimizer state to save
            epoch (int): Optional epoch number
            metadata (dict): Optional additional metadata
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_class": type(self).__name__,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if metadata is not None:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None, map_location=None):
        """
        Load model checkpoint and optionally restore optimizer state.

        Args:
            path (str): File path to load checkpoint from
            optimizer: Optional optimizer to restore state into
            map_location: Device to map loaded tensors to (e.g., 'cpu', 'cuda:0')

        Returns:
            dict: Checkpoint metadata (epoch, etc.)
        """
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["model_state_dict"])

        metadata = {}
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            metadata["epoch"] = checkpoint["epoch"]
        if "metadata" in checkpoint:
            metadata.update(checkpoint["metadata"])

        return metadata

    def count_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self):
        """Return the device this model is on."""
        return next(self.parameters()).device

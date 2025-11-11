"""Training utilities."""

import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

from .callbacks import EarlyStopping
from .evaluator import Evaluator


class Trainer:
    """Handles model training loop."""

    def __init__(
        self,
        model,
        criterion,
        device,
        model_name=None,
        log_dir="logs",
        lr=0.001,
        patience=3,
        window_size=None,
        offset=None,
        predictors=None,
        predict_features=None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.early_stopper = EarlyStopping(patience=patience, restore_best_weights=True)
        self.evaluator = Evaluator(criterion, device)
        self.history = {"loss": [], "val_loss": []}

        self.model_name = model_name or type(model).__name__

        # Build organized log directory structure: logs/ws{X}_offset{Y}/{model}/{predictors}_{prediction}
        if window_size is not None and offset is not None:
            run_config = f"ws{window_size}_offset{offset}"
            feature_combo = (
                f"{predictors}_{predict_features}"
                if predictors and predict_features
                else "default"
            )
            log_path = os.path.join(log_dir, run_config, self.model_name, feature_combo)
        else:
            # Fallback to simple structure if params not provided
            log_path = os.path.join(log_dir, self.model_name)

        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_path)

    def _plot_forecast(
        self, y_true, y_pred, variable_name="Variable", title="Original vs Predicted"
    ):
        """Return a matplotlib Figure comparing true vs predicted values."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y_true, label="Actual", marker="o", markersize=1)
        ax.plot(y_pred, label="Predicted", linestyle="--", marker="x", markersize=1)
        ax.set_title(title + f" ({variable_name})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel(variable_name)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

    def train_epoch(self, train_loader):
        """Train for one epoch, return average loss."""
        self.model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        return epoch_loss / len(train_loader.dataset)

    def fit(self, train_loader, val_loader, epochs, variable_name="Soil Moisture"):
        """Train model with validation and early stopping."""
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluator.evaluate(self.model, val_loader)
            val_loss = val_metrics["MSE"]

            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # --- TensorBoard logging ---
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Metrics/{k}", v, epoch)

            # Log a forecast visualization every few epochs
            # if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            self.model.eval()
            with torch.no_grad():
                X_sample, y_sample = next(iter(val_loader))
                X_sample, y_sample = X_sample.to(self.device), y_sample.to(self.device)
                y_pred = self.model(X_sample).detach().cpu().numpy().flatten()
                y_true = y_sample.detach().cpu().numpy().flatten()

                fig = self._plot_forecast(
                    y_true,
                    y_pred,
                    variable_name=variable_name,
                    title=f"{self.model_name.upper()} — Epoch {epoch+1}",
                )
                # Add to TensorBoard (works headlessly)
                self.writer.add_figure("Predicted_vs_Actual", fig, epoch + 1)
                plt.close(fig)
            if self.early_stopper(val_loss, self.model):
                break

        self.writer.flush()
        self.writer.close()

        return self.history

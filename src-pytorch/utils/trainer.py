"""Training utilities."""

import torch
import torch.optim as optim
from .callbacks import EarlyStopping
from .evaluator import Evaluator


class Trainer:
    """Handles model training loop."""

    def __init__(self, model, criterion, device, lr=0.001, patience=3):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.early_stopper = EarlyStopping(patience=patience, restore_best_weights=True)
        self.evaluator = Evaluator(criterion, device)
        self.history = {"loss": [], "val_loss": []}

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

    def fit(self, train_loader, val_loader, epochs):
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

            if self.early_stopper(val_loss, self.model):
                break

        return self.history

"""Training callbacks."""

import copy
import torch


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience=3, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("--- Early stopping triggered ---")
            if self.restore_best_weights and self.best_model_state is not None:
                print("Restoring best model weights.")
                model.load_state_dict(self.best_model_state)
            return True
        return False

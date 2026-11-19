"""Training loop with validation-based early stopping and full loss history.

Model selection uses the **validation** year only; the held-out test station-year
is never touched during training or early stopping, which keeps the final test
metric an unbiased estimate.
"""
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve a device string; ``auto`` picks CUDA when available, else CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            print("[warn] cuda requested but unavailable; using CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, *, epochs=40, patience=8,
                lr=1e-3, device=None):
    """Train with Adam + MSE, early-stopping on validation loss.

    Returns ``(model, history)`` where ``history`` is a dict with per-epoch
    ``train_loss`` / ``val_loss`` lists, ``epochs_completed``, ``best_val_epoch``,
    ``best_val_loss``, and ``final_val_loss``. The returned model is restored to
    its best-validation weights.
    """
    device = device or get_device()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    train_hist, val_hist = [], []
    best, best_state, bad, stopped = float("inf"), None, 0, epochs

    for ep in range(epochs):
        model.train()
        run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        train_hist.append(run / max(1, len(train_loader.dataset)))

        model.eval()
        run = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                run += crit(model(xb), yb).item() * len(xb)
        val_hist.append(run / max(1, len(val_loader.dataset)))

        if val_hist[-1] < best - 1e-7:
            best = val_hist[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                stopped = ep + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = dict(
        train_loss=train_hist, val_loss=val_hist,
        epochs_completed=stopped,
        best_val_epoch=int(np.argmin(val_hist)) + 1 if val_hist else -1,
        best_val_loss=best if best != float("inf") else float("nan"),
        final_val_loss=val_hist[-1] if val_hist else float("nan"),
        final_train_loss=train_hist[-1] if train_hist else float("nan"),
    )
    return model, history

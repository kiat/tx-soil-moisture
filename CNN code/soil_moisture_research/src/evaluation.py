"""Metrics and the end-to-end held-out station-year evaluation harness.

``compute_metrics`` and ``persistence_rmse`` are pure NumPy (no torch) so they
can be reused in analysis notebooks. ``run_fold`` ties the full pipeline together
(scaling -> windows -> train -> test) and needs torch.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from . import config as C
from .data_processing import make_windows, fold_keys


# ---------------------------------------------------------------------------
# Metrics (NumPy only)
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, and Pearson correlation on original SWC units."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return dict(RMSE=rmse, MAE=mae, CORR=corr)


def persistence_rmse(X_test, y_test) -> float:
    """RMSE of the persistence baseline (predict the window's last SWC value)."""
    last = X_test[:, -1, 0]  # SWC_20 is always feature column 0
    return float(np.sqrt(np.mean((y_test - last) ** 2)))


# ---------------------------------------------------------------------------
# Full fold evaluation (needs torch)
# ---------------------------------------------------------------------------
def run_fold(store, stations, test_station, test_year, feature_cols, *,
             window=24, horizon=72, arch="larger", seed=42,
             epochs=50, patience=8, batch=512, lr=1e-3, device=None, **model_kw):
    """Train on the fold and evaluate on the held-out (test_station, test_year).

    Returns ``(row, history)`` where ``row`` is a flat dict of metrics/metadata
    and ``history`` is the per-epoch loss history from ``train_model``.
    Metrics are computed in original SWC units (scaler inverse-transform).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
    from .models import build_model, count_params
    from .training import train_model, set_seed, get_device

    set_seed(seed)
    device = device or get_device()

    tr_k, va_k, te_k = fold_keys(stations, test_station, test_year)
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([store[k][feature_cols] for k in tr_k if k in store], axis=0))

    def collect(keys):
        Xs, ys = [], []
        for k in keys:
            if k not in store:
                continue
            g = store[k].copy()
            g[feature_cols] = scaler.transform(g[feature_cols])
            X, y = make_windows(g, feature_cols, window, horizon)
            if len(X):
                Xs.append(X)
                ys.append(y)
        if not Xs:
            return (np.empty((0, window, len(feature_cols)), "float32"),
                    np.empty((0,), "float32"))
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr, ytr = collect(tr_k)
    Xv, yv = collect(va_k)
    Xte, yte = collect(te_k)

    gen = torch.Generator().manual_seed(seed)

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                          generator=gen if shuffle else None)

    model = build_model(len(feature_cols), arch=arch, **model_kw)
    t0 = time.time()
    model, hist = train_model(model, loader(Xtr, ytr, True), loader(Xv, yv, False),
                              epochs=epochs, patience=patience, lr=lr, device=device)
    train_seconds = round(time.time() - t0, 1)

    # test predictions, inverse-transformed to SWC units
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader(Xte, yte, False):
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds).flatten() if preds else np.array([])

    dmin, dmax = scaler.data_min_[0], scaler.data_max_[0]
    inv = lambda v: v * (dmax - dmin) + dmin
    act, prd = inv(yte), inv(preds)

    m = compute_metrics(act, prd)
    p_rmse = float(np.sqrt(np.mean((act - inv(Xte[:, -1, 0])) ** 2)))
    row = dict(
        test_station=test_station, test_year=test_year, horizon=horizon,
        features="|".join(feature_cols), n_features=len(feature_cols),
        arch=arch, window=window, seed=seed, n_test=len(yte),
        RMSE=m["RMSE"], MAE=m["MAE"], CORR=m["CORR"],
        persistence_rmse=p_rmse,
        impr_vs_persist_pct=100 * (p_rmse - m["RMSE"]) / p_rmse if p_rmse else float("nan"),
        n_params=count_params(model),
        epochs_completed=hist["epochs_completed"], best_val_epoch=hist["best_val_epoch"],
        best_val_loss=hist["best_val_loss"], final_val_loss=hist["final_val_loss"],
        train_seconds=train_seconds,
    )
    return row, hist

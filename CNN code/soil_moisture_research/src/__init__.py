"""Reusable pipeline for TxSON CNN soil-moisture forecasting.

Modules
-------
config          paths, stations, splits, feature groups
features        causal feature engineering + precipitation-lag correlation
data_processing loading, windowing, held-out station-year folds
models          SmallCNN / LargerCNN / ConfigurableCNN
training        seeded training loop with early stopping + loss history
evaluation      metrics, persistence baseline, full fold harness
"""
from . import config, features, data_processing  # noqa: F401

__all__ = ["config", "features", "data_processing", "models",
           "training", "evaluation"]

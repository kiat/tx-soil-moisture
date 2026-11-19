"""Data loading, window construction, and held-out station-year splits.

This module has **no torch dependency** so it can be imported for data audits
and lag analysis on a machine without a deep-learning stack installed.
"""
from __future__ import annotations
import os
import urllib.request
import numpy as np
import pandas as pd

from . import config as C
from .features import engineer_causal


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataframe(path: str | None = None, download: bool = True) -> pd.DataFrame:
    """Load the TxSON33 parquet, downloading it to ``data/`` if necessary."""
    path = path or C.find_parquet()
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"[data] downloading dataset -> {path}")
        urllib.request.urlretrieve(C.DATA_URL, path)
    df = pd.read_parquet(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["year"] = df["Timestamp"].dt.year
    return df


def build_station_year_store(df: pd.DataFrame,
                             stations: list[str] | None = None) -> dict:
    """Return ``{(station, year): engineered_frame}`` for the requested stations.

    Feature engineering is done *per (station, year)* so rolling/lagged features
    never span a year or station boundary.
    """
    stations = stations or C.SELECTED_STATIONS
    store = {}
    for s in stations:
        sdf = df[df["Station"] == s]
        for y in C.ALL_YEARS:
            g = sdf[sdf["year"] == y]
            if len(g):
                store[(s, y)] = engineer_causal(g)
    return store


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def make_windows(g: pd.DataFrame, feature_cols: list[str],
                 window: int, horizon: int):
    """Slice one (station, year) frame into (X, y) supervised windows.

    ``X[i]`` is a ``window``-hour block of features; ``y[i]`` is ``SWC_20``
    ``horizon`` hours after the block's last hour. Windows never cross the frame
    boundary, so no window spans a year or station change.
    """
    X = g[feature_cols].to_numpy("float32")
    y = g[C.TARGET].to_numpy("float32")
    n = len(g) - window - horizon + 1
    if n <= 0:
        return (np.empty((0, window, len(feature_cols)), "float32"),
                np.empty((0,), "float32"))
    Xs = np.stack([X[i:i + window] for i in range(n)])
    ys = np.array([y[i + window + horizon - 1] for i in range(n)], "float32")
    return Xs, ys


# ---------------------------------------------------------------------------
# Held-out station-year fold definition
# ---------------------------------------------------------------------------
def fold_keys(stations: list[str], test_station: str, test_year: int):
    """Split (station, year) keys into train / validation / test lists.

    * test  = the single held-out (test_station, test_year)
    * val   = VAL_YEAR of every *other* station (never the test station)
    * train = everything else (includes other years of the test station)
    """
    train, val, test = [], [], [(test_station, test_year)]
    for s in stations:
        for y in C.ALL_YEARS:
            if s == test_station and y == test_year:
                continue
            if y == C.VAL_YEAR and s != test_station:
                val.append((s, y))
            else:
                train.append((s, y))
    return train, val, test

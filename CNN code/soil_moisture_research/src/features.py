"""Causal feature engineering for the TxSON soil-moisture pipeline.

Every engineered feature uses **only past or current** observations, so a feature
value at hour ``t`` never depends on data after ``t``. This is what makes the
features safe to use as forecasting inputs (no future leakage). Specifically:

* ``Ppt_log``          : log(1 + Ppt), current hour only.
* ``Ppt_6h/24h/72h``   : trailing rolling sums (backward-looking).
* ``Ppt_log_lagL``     : precipitation shifted forward by L hours (past values).
* ``hours_since_rain`` : counter reset on rain (depends only on the past).
* ``Tair/RH/...``      : forward-filled (never back-filled).
* ``sin_doy/cos_doy``  : deterministic function of the timestamp.

No ``bfill`` / backward fill is ever applied, because backward filling copies a
*future* observation into a *past* gap and would leak information.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Lags (hours) used for the "lagged rain" configuration and lag diagnostics.
RAIN_LAGS = (1, 2, 3, 6, 12, 24)


def engineer_causal(g: pd.DataFrame) -> pd.DataFrame:
    """Add engineered feature columns to a single (station, year) frame.

    Parameters
    ----------
    g : DataFrame with at least ``Timestamp`` and the raw TxSON columns for one
        station and one calendar year.

    Returns
    -------
    DataFrame sorted by time with the engineered columns added.
    """
    g = g.sort_values("Timestamp").reset_index(drop=True).copy()

    # --- precipitation -----------------------------------------------------
    g["Ppt"] = g["Ppt"].fillna(0.0).clip(lower=0.0)   # defensive: no negatives
    g["Ppt_log"] = np.log1p(g["Ppt"])                 # compress the long tail

    # trailing (causal) rolling accumulations
    g["Ppt_6h"] = g["Ppt"].rolling(6, min_periods=1).sum()
    g["Ppt_24h"] = g["Ppt"].rolling(24, min_periods=1).sum()
    g["Ppt_72h"] = g["Ppt"].rolling(72, min_periods=1).sum()

    # hours since last measurable rain (resets to 0 on a rainy hour)
    rainy = (g["Ppt"] > 0.01).astype(int)
    g["hours_since_rain"] = rainy.groupby((rainy != rainy.shift()).cumsum()).cumcount()
    g.loc[g["Ppt"] > 0.01, "hours_since_rain"] = 0
    g["rainy_hours_24h"] = rainy.rolling(24, min_periods=1).sum()

    # causal lagged precipitation (shift copies past values forward)
    for L in RAIN_LAGS:
        g[f"Ppt_log_lag{L}"] = g["Ppt_log"].shift(L).fillna(0.0).astype("float32")

    # --- meteorology (forward-fill only; never backward) -------------------
    for c in ["Tair", "RH", "Srad", "T_20", "Wind speed", "Wind direction"]:
        if c in g.columns:
            s = g[c].ffill()
            s = s.fillna(s.expanding().mean()).fillna(0.0)
            g[c] = s

    # wind vector components
    if "Wind speed" in g.columns and "Wind direction" in g.columns:
        wd = np.deg2rad(g["Wind direction"].values)
        g["Wind_u"] = (g["Wind speed"].values * np.sin(wd)).astype("float32")
        g["Wind_v"] = (g["Wind speed"].values * np.cos(wd)).astype("float32")

    # --- seasonality (deterministic in the timestamp) ----------------------
    doy = g["Timestamp"].dt.dayofyear
    g["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    g["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")

    return g


def precip_lag_correlation(g: pd.DataFrame, target: str = "SWC_20",
                           lags=(0, 1, 2, 3, 6, 12, 24, 48, 72)) -> pd.DataFrame:
    """Correlate rainfall at time t with the soil-moisture *change* at t+lag.

    Two correlation targets are reported for each lag:

    * ``corr_instantaneous`` : corr( Ppt[t], dSWC[t+lag] ) where
      ``dSWC[k] = SWC[k] - SWC[k-1]`` -- the moisture change in the single hour
      ``lag`` hours after the rain.
    * ``corr_cumulative``    : corr( Ppt[t], SWC[t+lag] - SWC[t] ) -- the total
      moisture rise accumulated from the rain hour out to ``lag`` hours later.

    Returns a tidy DataFrame with one row per lag.
    """
    g = g.sort_values("Timestamp").reset_index(drop=True)
    ppt = g["Ppt"].fillna(0.0).to_numpy()
    swc = g[target].to_numpy()
    dswc = np.concatenate([[np.nan], np.diff(swc)])  # hourly change
    rows = []
    for lag in lags:
        # instantaneous: rain at t vs the one-hour change at t+lag
        if lag == 0:
            inst = _safe_corr(ppt, dswc)
            cum = np.nan
        else:
            inst = _safe_corr(ppt[:-lag], dswc[lag:])
            cum = _safe_corr(ppt[:-lag], swc[lag:] - swc[:-lag])
        rows.append(dict(lag_h=lag, corr_instantaneous=inst, corr_cumulative=cum))
    return pd.DataFrame(rows)


def _safe_corr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])

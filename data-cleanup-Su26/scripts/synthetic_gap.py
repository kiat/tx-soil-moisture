"""Punch a synthetic NaN gap into a station DataFrame, for testing imputation methods."""
import numpy as np
import pandas as pd


def _n_rows(value, step, name):
    """Turn a row count, or a time span like '36h', into a whole number of rows."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    span = pd.Timedelta(value)
    if span % step != pd.Timedelta(0):
        raise ValueError(f"{name} {span} is not a multiple of the sampling step {step}")
    return span // step


def synthetic_gap(df, length, column=None, start=None, seed=None, context=None):
    """Return (gapped, answer): a copy of df with one artificial NaN gap, plus the answer key.

    df      : DataFrame with a DatetimeIndex (a prewashed station).
    length  : gap size — a row count, or a time span like '36h' or '3D'.
    column  : one column name, or a list of them; None will gap every measurement
              column. 'Flag' is a QC flag, not a measurement, so it is never gapped.
    start   : timestamp of the first missing row; None picks a random window that
              doesn't overlap any real gap, so the answer key is complete.
    seed    : seed for the random placement, for reproducible experiments.
    context : how much data to keep on each side of the gap (rows, or a span like
              '7D'); None returns the full frame.

    Score an imputation with e.g. `imputed.loc[answer.index, answer.columns]`
    against `answer`.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex (run the prewash first).")

    step = pd.Timedelta(np.median(np.diff(df.index.values)))   # sampling interval, e.g. 1h

    # gap size in rows
    n = _n_rows(length, step, "length")
    if n < 1:
        raise ValueError("length must be at least one row")
    if n > len(df):
        raise ValueError(f"a {n}-row gap does not fit in {len(df)} rows of data")

    # which columns to knock out
    if column is None:
        cols = list(df.columns)
    elif isinstance(column, str):
        cols = [column]
    else:
        cols = list(column)
    cols = [c for c in cols if c != "Flag"]                    # never gap the flag
    if not cols:
        raise ValueError("no columns to gap ('Flag' is a QC flag and is never gapped)")

    # rows usable as ground truth: no real NaN in any target column
    clean = df[cols].notna().all(axis=1).to_numpy()

    # find i, the row where the gap starts
    if start is not None:
        start = pd.Timestamp(start)
        if start not in df.index:
            raise ValueError(f"start {start} is not in the index")
        i = df.index.get_loc(start)
        if i + n > len(df):
            raise ValueError(f"a {n}-row gap starting at {start} runs past the end of the data")
        if not clean[i:i + n].all():
            raise ValueError(f"the window at {start} overlaps a real gap, so there is no ground truth")
    else:
        # count the clean rows in each n-row window, then keep the all-clean windows
        clean_in_window = np.convolve(clean, np.ones(n, dtype=int), mode="valid") # vectorization by AI
        candidates = np.flatnonzero(clean_in_window == n) # of the clean windows, which have length = n ?
        if len(candidates) == 0:
            raise ValueError(f"no clean window of {n} rows exists in columns {cols}")
        i = int(np.random.default_rng(seed).choice(candidates))

    # save the answer key, then punch the hole (mask() keeps column dtypes safe)
    answer = df.iloc[i:i + n][cols].copy()
    hole = np.zeros(len(df), dtype=bool)
    hole[i:i + n] = True
    gapped = df.copy()
    for col in cols:
        gapped[col] = gapped[col].mask(hole)

    # optionally trim to just the neighbourhood of the gap
    if context is not None:
        pad = _n_rows(context, step, "context")
        if pad < 0:
            raise ValueError("context must not be negative")
        gapped = gapped.iloc[max(0, i - pad):i + n + pad]

    return gapped, answer

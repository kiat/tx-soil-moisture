"""Punch a synthetic NaN gap into a station DataFrame, for testing imputation methods."""
import numpy as np
import pandas as pd


def _n_rows(value, step, name):
    """Convert a row count or Timedelta-like value into a whole number of rows."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    span = pd.Timedelta(value)
    n, rem = divmod(span, step)
    if rem != pd.Timedelta(0):
        raise ValueError(f"{name} {span} is not a multiple of the sampling step {step}")
    return n


def synthetic_gap(df, length, column=None, start=None, seed=None, context=None):
    """Return a copy of df with one artificial NaN gap, plus the answer key.

    df      : DataFrame with a DatetimeIndex (a prewashed station).
    length  : gap size — int (number of rows) or anything pd.Timedelta accepts
              (e.g. '36h', '3D'); must be a whole multiple of the sampling step.
    column  : restrict the gap to one column (or a list of columns); None knocks
              out every measurement column. 'Flag' is a QC flag, not a
              measurement, so it is never gapped and is ignored in selections.
    start   : timestamp of the first missing row; None picks a random window
              that doesn't overlap any real gap, so the answer key is complete.
    seed    : seed for the random placement, for reproducible experiments.
    context : trim the returned frame to this much data on each side of the gap
              (rows or a Timedelta like '7D'); None returns the full frame.

    Returns (gapped, answer):
    gapped : copy of df with the gap punched in.
    answer : the original values over the gap window (ground truth). Score an
             imputation with e.g. `imputed.loc[answer.index, answer.columns]`
             against `answer`.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex (run the prewash first).")

    step = pd.Timedelta(np.median(np.diff(df.index.values)))   # sampling interval

    n = _n_rows(length, step, "length")                        # gap length in rows
    if n < 1:
        raise ValueError("length must be at least one row")
    if n > len(df):
        raise ValueError(f"a {n}-row gap does not fit in {len(df)} rows of data")

    if column is None:
        cols = list(df.columns)
    elif isinstance(column, str):
        cols = [column]
    else:
        cols = list(column)
    cols = [c for c in cols if c != "Flag"]                    # never gap the QC flag
    if not cols:
        raise ValueError("no columns to gap ('Flag' is a QC flag and is never gapped)")

    # rows usable as ground truth: no real NaN in any target column
    clean = df[cols].notna().all(axis=1).to_numpy()

    # handle if start is provided
    if start is not None:
        start = pd.Timestamp(start)
        try:
            i = df.index.get_loc(start)
        except KeyError:
            raise ValueError(f"start {start} is not in the index")
        if i + n > len(df):
            raise ValueError(f"a {n}-row gap starting at {start} runs past the end of the data")
        if not clean[i:i + n].all():
            raise ValueError(f"the window at {start} overlaps a real gap, so there is no ground truth")
    else:
        # positions where the next n rows are all clean
        ok = np.convolve(clean, np.ones(n, dtype=int), mode="valid") == n
        candidates = np.flatnonzero(ok)
        if len(candidates) == 0:
            raise ValueError(f"no clean window of {n} rows exists in columns {cols}")
        i = int(np.random.default_rng(seed).choice(candidates))

    answer = df.iloc[i:i + n][cols].copy()

    hole = np.zeros(len(df), dtype=bool)
    hole[i:i + n] = True
    gapped = df.copy()
    for col in cols:
        gapped[col] = gapped[col].mask(hole)    # mask, not iloc: upcasts int columns to float

    if context is not None:
        pad = _n_rows(context, step, "context")
        if pad < 0:
            raise ValueError("context must not be negative")
        gapped = gapped.iloc[max(0, i - pad):i + n + pad]

    return gapped, answer
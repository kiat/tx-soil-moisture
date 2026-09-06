"""Shared timestamp invariants for staged pipeline data."""
from __future__ import annotations

import pandas as pd


def require_unique_datetime_index(
    df: pd.DataFrame,
    context: str,
) -> pd.DataFrame:
    """Fail when a post-Stage-0 time series contains duplicate timestamps."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{context} must use a DatetimeIndex.")
    if not df.index.has_duplicates:
        return df

    duplicate_index = df.index[df.index.duplicated(keep=False)]
    timestamps = duplicate_index.unique().sort_values()
    preview = ", ".join(str(value) for value in timestamps[:5])
    suffix = " ..." if len(timestamps) > 5 else ""
    raise ValueError(
        f"{context} violates the post-Stage-0 invariant: "
        f"{len(timestamps)} duplicate timestamp(s) remain ({preview}{suffix}). "
        "Resolve duplicates during datacleaning.py before continuing."
    )

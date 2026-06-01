import pandas as pd
import numpy as np
 
 
def time_cleaner(df, avg_diff_measurements=False, extend_time=False):
    """Clean timestamp issues in a time-indexed DataFrame.
 
    Two cases are handled:
      1. Gaps      -> missing timestamps are inserted on a regular hourly
                      grid (new rows are left as NaN).
      2. Collisions -> rows that share a timestamp but hold different
                      measurements. Exact duplicates are assumed to have
                      already been removed by dup_cleaner.py.
 
    A collision is only resolved if a strategy is chosen:
      - avg_diff_measurements: collapse the colliding rows into one averaged row.
      - extend_time:           keep the row closest to the previous measurement
                               at the original time, push the rest into the
                               following slots, and shift later rows forward so
                               nothing overlaps.
 
    Expects `df.index` to be a DatetimeIndex (RECORD / flag columns dropped).
    """
    if avg_diff_measurements and extend_time:
        raise ValueError("Pick only one of avg_diff_measurements or extend_time.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex (run dup_cleaner.py first).")
    if df.empty:
        return df

    # warn about non numeric columns that will be ignored when averaging and from measure of distance for extending timestamps
    if avg_diff_measurements or extend_time:
        non_numeric = df.select_dtypes(exclude="number").columns
        if len(non_numeric):
            print(
                "Warning: the following non-numeric columns will be ignored when "
                "averaging colliding rows and measuring distances for extending timestamps: "
                + ", ".join(non_numeric)
            )
 
    # --- case 2: same timestamp, different measurements ---
    if df.index.has_duplicates:
        if avg_diff_measurements:
            df = df.groupby(df.index).mean(numeric_only=True)
        elif extend_time:
            df = _extend_duplicate_timestamps(df)
        else:
            raise ValueError(
                "Duplicate timestamps found; set avg_diff_measurements=True or "
                "extend_time=True to resolve them."
            )
 
    # --- case 1: insert missing timestamps ---
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='h')
    return df.reindex(full_range)
 
 
def _extend_duplicate_timestamps(df):
    """Resolve duplicate timestamps: keep one row at its original time and bump
    the rest into later hourly slots, shifting all subsequent rows forward so the
    index stays unique and ordered. Within a collision, the row closest (in numeric
    space) to the previous kept row is placed first."""

    print("Extending duplicate timestamps...")

    df = df.sort_index(kind="stable") # sort by time, but keep original order within duplicates (important for measuring distance to previous row)

    # input check
    numeric = df.select_dtypes("number")
    if numeric.empty:
        print("Warning: no numeric columns; duplicates keep their original order.")

    # get the timestamps as unique indices (in original order) as we will be changing timestamps 
    index_blocks = df.groupby(df.index, sort=False).indices.values()

    # Reorder each block of indices by similarity to the previous kept row.
    order = []
    for indices in index_blocks:
        indices = list(indices)
        if len(indices) > 1 and order and not numeric.empty:
            dist = (numeric.iloc[indices] - numeric.iloc[order[-1]]).abs().sum(axis=1)
            indices = [indices[i] for i in dist.to_numpy().argsort(kind="stable")]
        order.extend(indices) # extend, not append, because indices can have more than one element ( three duplicates would have two indices that need to be added to the order list )

    # New time = original + (global position − group position) hours.
    df = df.iloc[order].copy()
    df.index = df.index + pd.to_timedelta(np.arange(len(df)) - pd.factorize(df.index)[0], unit="h")
    return df

    
"""Count gaps in each column of a dataframe, including the timestamp index."""
import numpy as np
import pandas as pd

LABELS = ["<24h", "1-7d", "7-30d", ">30d"]


def bucket(span):
    """Duration bucket for a single gap."""
    day = pd.Timedelta(days=1)
    if span < day:
        return "<24h"
    if span < 7 * day:
        return "1-7d"
    if span <= 30 * day:
        return "7-30d"
    return ">30d"


def _count(spans):
    """Tally gap durations into buckets."""
    counts = {label: 0 for label in LABELS}
    for span in spans:
        counts[bucket(span)] += 1
    return counts


def _sampling_step(df):
    """The typical time between rows, e.g. 1h for the prewashed stations."""
    return pd.Timedelta(np.median(np.diff(df.index.values)))


def _na_runs(series):
    """Yield (start, end) index labels for each run of consecutive NaNs."""
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()   # label each run of equal values
    for _, run in missing.groupby(run_id):
        if run.iloc[0]:                              # a NaN run, not a run of data
            yield run.index[0], run.index[-1]


def gap_report(df):
    """Print and return gap counts by duration for the timestamp index and each column.

    Expects the timestamp index to be a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("timestamp column must be set as index before running gap_report.py.")

    step = _sampling_step(df)

    # gaps in the index itself: a jump bigger than one step means timestamps are
    # missing there, and the missing span is the jump minus the one expected step
    jumps = df.index.to_series().diff()
    rows = {"timestamps": _count(jumps[jumps > step] - step)}

    # gaps in each column: runs of consecutive NaNs
    for col in df.columns:
        spans = [end - start + step for start, end in _na_runs(df[col])]
        rows[col] = _count(spans)

    report = pd.DataFrame(rows).T[LABELS]
    report.index.name = "gaps_in"
    print(report)
    return report


def gap_durations(df, group=None, column=None):
    """List the individual NA-gap lengths in a time-indexed DataFrame.

    df     : DataFrame with a DatetimeIndex.
    group  : one of LABELS ('<24h', '1-7d', '7-30d', '>30d'); None keeps every bucket.
    column : restrict to a single column; None scans all columns.

    Returns a DataFrame with one row per gap: [column, start, end, duration, group],
    sorted shortest to longest. Take the 'duration' column for just the lengths.
    """
    if group is not None and group not in LABELS:
        raise ValueError(f"group must be one of {LABELS} or None")

    step = _sampling_step(df)
    cols = [column] if column is not None else list(df.columns)
    rows = []
    for col in cols:
        for start, end in _na_runs(df[col]):
            duration = end - start + step   # a 1-row gap starts and ends on the same
            label = bucket(duration)        # timestamp, so add one step to its length
            if group is None or label == group:
                rows.append((col, start, end, duration, label))

    out = pd.DataFrame(rows, columns=["column", "start", "end", "duration", "group"])
    return out.sort_values("duration").reset_index(drop=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Report gaps (missing timestamps and consecutive-NaN runs), "
                    "bucketed by duration, in a TxSON .dat file."
    )
    parser.add_argument("input_file", help="prewashed csv file to report on")
    parser.add_argument("output_file", nargs="?", default=None,
                        help="optional path to write the gap report as a CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, parse_dates=[0], index_col=0)
    report = gap_report(df)

    if args.output_file:
        report.to_csv(args.output_file)
        print(f"\ngap report written to {args.output_file}\n")


if __name__ == "__main__":
    main()

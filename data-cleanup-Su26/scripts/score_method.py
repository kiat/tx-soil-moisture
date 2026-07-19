"""Score imputation methods on the synthetic-gap samples drawn by gap_sampler.
Library-only for now (no CLI)."""
import numpy as np
import pandas as pd


def score_method(impute, samples, bins=None, name=None):
    """Score an impute(gapped, column) -> Series function on every synthetic gap.

    impute  : function taking (gapped, column) and returning the column as a
              Series with the gap filled (extra filled values outside the gap
              are fine — only the gap window is scored).
    samples : list of sample dicts from gap_sampler.sample_gaps.
    bins    : None -> one category per distinct gap length (labels like "6h"),
              matching gap_sampler's exact-length draws; or bin edges in hours,
              e.g. [0, 6, 12, 24], to group lengths into ranges (labels like
              "1-6h") — useful when the sample holds many distinct lengths.
    name    : when given (e.g. "nearest observation"), print a
              short report: the name, the overall RMSE / MAE / bias, and a
              length_bin x rain_in_gap summary table. None -> stay silent.

    Returns one row per gap, as a DataFrame with columns:

    station     : str — station the gap was drawn from.
    column      : str — the SWC column that was gapped.
    length_h    : int — gap length in hours.
    rain_in_gap : bool — True if Ppt > 0 at any hour inside the gap window.
    rmse        : float — root-mean-square error of the fill over the gap.
    mae         : float — mean absolute error of the fill over the gap.
    bias        : float — mean error (fill - truth); positive = fill too wet.
    length_bin  : category — the gap's length group (see `bins`).
    """
    rows = []
    for s in samples:
        est   = impute(s["gapped"], s["column"]).loc[s["answer"].index]
        truth = s["answer"][s["column"]]
        err   = est - truth
        rows.append({"station": s["station"], "column": s["column"],
                     "length_h": s["length_h"],
                     "rain_in_gap": bool(s["gapped"].loc[s["answer"].index, "Ppt"]
                                         .fillna(0).gt(0).any()),
                     "rmse": float(np.sqrt((err ** 2).mean())),
                     "mae":  float(err.abs().mean()),
                     "bias": float(err.mean())})
    scores = pd.DataFrame(rows)
    if bins is None:
        # one category per distinct length, in length order
        order = [f"{h}h" for h in sorted(scores["length_h"].unique())]
        scores["length_bin"] = pd.Categorical([f"{h}h" for h in scores["length_h"]],
                                              categories=order, ordered=True)
    else:
        labels = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            if int(hi) == 24:
                hi = 23          # a bin ending at 24 holds the <24h gaps, whose max is 23h
            labels.append(f"{int(lo) + 1}-{int(hi)}h")
        scores["length_bin"] = pd.cut(scores["length_h"], bins, labels=labels)

    if name is not None:
        print(f"{name}, {len(scores)} gaps (m3/m3)")
        print(f"overall: RMSE {scores['rmse'].mean():.4f}   MAE {scores['mae'].mean():.4f}   "
              f"bias {scores['bias'].mean():+.4f}")
        summary = (scores.groupby(["length_bin", "rain_in_gap"], observed=True)
                         .agg(n=("rmse", "size"), rmse=("rmse", "mean"),
                              mae=("mae", "mean"), bias=("bias", "mean"))
                         .reset_index().round(4))
        print(summary.to_string(index=False))

    return scores

"""Fill remaining soil NaNs after validation and sensor-level QC.

This is a final non-destructive filling stage. It reads the sensor-QC output
when available:

    output/Station{site}_filled_sensor_qc.csv

and writes:

    output/Station{site}_filled_final.csv

For station/parameter pairs with enough remaining observed target data, the
script uses the best correlated donor station and a linear donor regression.
When no target training data are available, which happens for sensor columns
masked by sensor-level QC, it falls back to the hourly mean of usable donor
stations and records that lower-confidence method in the detail log.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from param_config import ALL_SOIL_PARAMS


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
DEFAULT_PARAMS = ALL_SOIL_PARAMS


def input_path_for(station: str) -> Path:
    candidates = [
        OUT_DIR / f"Station{station}_filled_sensor_qc.csv",
        OUT_DIR / f"Station{station}_filled_verylonggaps_repaired.csv",
        OUT_DIR / f"Station{station}_filled_verylonggaps.csv",
        OUT_DIR / f"Station{station}_filled_longgaps_repaired.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def discover_stations() -> List[str]:
    pat = re.compile(r"Station(.+)_filled_sensor_qc\.csv")
    stations = [
        m.group(1)
        for path in OUT_DIR.glob("Station*_filled_sensor_qc.csv")
        if (m := pat.match(path.name))
    ]
    if stations:
        return sorted(stations)

    fallback = re.compile(r"Station(.+)_filled_verylonggaps_repaired\.csv")
    return sorted(
        m.group(1)
        for path in OUT_DIR.glob("Station*_filled_verylonggaps_repaired.csv")
        if (m := fallback.match(path.name))
    )


def read_station(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    return ensure_hourly_regular_index(df)


def ensure_hourly_regular_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.empty:
        return df
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(full_idx)


def min_std_for(param: str) -> float:
    if param.startswith("SWC_"):
        return 1e-4
    if param.startswith("T_"):
        return 0.05
    return 0.0


def usable_series(series: pd.Series, min_std: float, min_count: int = 1) -> bool:
    s = series.dropna()
    return len(s) >= min_count and float(s.std()) >= min_std


def choose_best_donor(
    target: pd.Series,
    donor_series: Dict[str, pd.Series],
    min_overlap: int,
    min_abs_corr: float,
    min_std: float,
) -> Tuple[Optional[str], float, int]:
    if not usable_series(target, min_std, min_overlap):
        return None, float("nan"), 0

    best_sid, best_corr, best_overlap = None, -np.inf, 0
    for sid, donor in donor_series.items():
        if not usable_series(donor, min_std, min_overlap):
            continue
        mask = target.notna() & donor.notna()
        overlap = int(mask.sum())
        if overlap < min_overlap:
            continue
        corr = target[mask].corr(donor[mask])
        if pd.isna(corr) or abs(corr) < min_abs_corr:
            continue
        if abs(corr) > best_corr:
            best_sid = sid
            best_corr = abs(float(corr))
            best_overlap = overlap
    return best_sid, best_corr, best_overlap


def fit_linear_map(target: pd.Series, donor: pd.Series) -> LinearRegression:
    mask = target.notna() & donor.notna()
    return LinearRegression().fit(donor[mask].values.reshape(-1, 1), target[mask])


def linear_prediction(idx: pd.DatetimeIndex, donor: pd.Series, model: LinearRegression) -> pd.Series:
    x = donor.reindex(idx).dropna()
    if x.empty:
        return pd.Series(dtype=float)
    return pd.Series(model.predict(x.values.reshape(-1, 1)), index=x.index)


def donor_mean_prediction(
    idx: pd.DatetimeIndex,
    donors: Dict[str, pd.DataFrame],
    param: str,
    min_std: float,
) -> pd.Series:
    donor_series = {
        sid: df[param].reindex(idx)
        for sid, df in donors.items()
        if param in df.columns and usable_series(df[param], min_std, min_count=24)
    }
    if not donor_series:
        return pd.Series(dtype=float)
    return pd.DataFrame(donor_series).mean(axis=1, skipna=True).dropna()


def donor_climatology_prediction(
    idx: pd.DatetimeIndex,
    donors: Dict[str, pd.DataFrame],
    param: str,
    min_std: float,
) -> pd.Series:
    """Fallback for timestamps outside donor coverage.

    Uses donor values from the same day-of-year and hour across available
    years. If that exact seasonal-hour bin is unavailable, falls back to the
    day-of-year mean, then the global donor mean.
    """
    frames: List[pd.DataFrame] = []
    for sid, df in donors.items():
        if param not in df.columns or not usable_series(df[param], min_std, min_count=24):
            continue
        s = df[param].dropna()
        if s.empty:
            continue
        frame = pd.DataFrame(
            {
                "value": s.astype(float),
                "doy": s.index.dayofyear,
                "hour": s.index.hour,
            }
        )
        frames.append(frame)

    if not frames:
        return pd.Series(dtype=float)

    samples = pd.concat(frames, ignore_index=True)
    by_doy_hour = samples.groupby(["doy", "hour"])["value"].mean()
    by_doy = samples.groupby("doy")["value"].mean()
    global_mean = float(samples["value"].mean())

    values = []
    out_index = []
    for ts in idx:
        key = (ts.dayofyear, ts.hour)
        if key in by_doy_hour.index:
            val = float(by_doy_hour.loc[key])
        elif ts.dayofyear in by_doy.index:
            val = float(by_doy.loc[ts.dayofyear])
        else:
            val = global_mean
        values.append(val)
        out_index.append(ts)
    return pd.Series(values, index=pd.DatetimeIndex(out_index))


def correct_boundary_drift(preds: pd.Series, observed: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if preds.empty:
        return preds
    left_ts = start - timedelta(hours=1)
    right_ts = end + timedelta(hours=1)
    has_left = left_ts in observed.index and pd.notna(observed.loc[left_ts])
    has_right = right_ts in observed.index and pd.notna(observed.loc[right_ts])

    corrected = preds.copy()
    if has_left and has_right:
        left_delta = observed.loc[left_ts] - corrected.iloc[0]
        right_delta = observed.loc[right_ts] - corrected.iloc[-1]
        if len(corrected) == 1:
            corrected.iloc[0] = 0.5 * (observed.loc[left_ts] + observed.loc[right_ts])
        else:
            corrected = corrected + np.linspace(left_delta, right_delta, len(corrected))
    elif has_left:
        corrected.iloc[0] = 0.5 * (observed.loc[left_ts] + corrected.iloc[0])
    elif has_right:
        corrected.iloc[-1] = 0.5 * (observed.loc[right_ts] + corrected.iloc[-1])
    return corrected


def apply_physical_bounds(values: pd.Series, param: str) -> pd.Series:
    if param.startswith("SWC_"):
        return values.clip(lower=0.0, upper=0.6)
    if param.startswith("T_"):
        return values.clip(lower=-30.0, upper=60.0)
    return values


def nan_runs(df: pd.DataFrame, param: str) -> List[pd.DatetimeIndex]:
    if param not in df.columns:
        return []
    mask = df[param].isna().to_numpy()
    runs: List[pd.DatetimeIndex] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        runs.append(pd.DatetimeIndex(df.index[i:j]))
        i = j
    return runs


def fill_station(
    station: str,
    params: Iterable[str],
    all_data: Dict[str, pd.DataFrame],
    min_overlap: int,
    min_abs_corr: float,
) -> None:
    print(f"\n=== Station {station} | final residual filling ===")
    target_df = all_data[station].copy()
    donors = {sid: df for sid, df in all_data.items() if sid != station}
    detail_rows: List[Dict[str, object]] = []

    for param in params:
        if param not in target_df.columns:
            print(f"  {param}: column missing, skip.")
            continue

        runs = nan_runs(target_df, param)
        if not runs:
            print(f"  {param}: no residual NaN.")
            continue

        available_donors = {sid: df for sid, df in donors.items() if param in df.columns}
        if not available_donors:
            print(f"  {param}: no donor columns, skip.")
            continue

        min_std = min_std_for(param)
        observed_count = int(target_df[param].notna().sum())
        donor_sid, corr, overlap = choose_best_donor(
            target_df[param],
            {sid: df[param] for sid, df in available_donors.items()},
            min_overlap=min_overlap,
            min_abs_corr=min_abs_corr,
            min_std=min_std,
        )
        model = None
        if donor_sid is not None:
            model = fit_linear_map(target_df[param], available_donors[donor_sid][param])
            print(f"  {param}: {len(runs)} run(s), donor={donor_sid}, |r|={corr:.3f}, overlap={overlap}")
        else:
            print(f"  {param}: {len(runs)} run(s), no regression donor; donor-mean fallback")

        filled_count = 0
        for idx in runs:
            start, end = idx[0], idx[-1]
            pred_parts: List[pd.DataFrame] = []
            linear_preds = pd.Series(dtype=float)

            if model is not None and donor_sid is not None:
                linear_preds = linear_prediction(idx, available_donors[donor_sid][param], model).dropna()
                if not linear_preds.empty:
                    pred_parts.append(pd.DataFrame({
                        "Filled": linear_preds,
                        "Method": "linear_donor",
                        "Donor": donor_sid,
                        "Abs Corr": corr,
                        "Overlap Hours": overlap,
                    }))

            fallback_idx = idx.difference(linear_preds.index)
            if len(fallback_idx) > 0:
                mean_preds = donor_mean_prediction(fallback_idx, available_donors, param, min_std).dropna()
                if not mean_preds.empty:
                    method = (
                        "donor_mean_no_target_training"
                        if observed_count < min_overlap
                        else "donor_mean_missing_linear_donor"
                    )
                    pred_parts.append(pd.DataFrame({
                        "Filled": mean_preds,
                        "Method": method,
                        "Donor": np.nan,
                        "Abs Corr": np.nan,
                        "Overlap Hours": observed_count,
                    }))

            predicted_idx = pd.DatetimeIndex([])
            if pred_parts:
                predicted_idx = pd.DatetimeIndex(pd.concat(pred_parts).index.unique())
            climatology_idx = idx.difference(predicted_idx)
            if len(climatology_idx) > 0:
                clim_preds = donor_climatology_prediction(climatology_idx, available_donors, param, min_std).dropna()
                if not clim_preds.empty:
                    pred_parts.append(pd.DataFrame({
                        "Filled": clim_preds,
                        "Method": "donor_climatology_no_timestamp_donor",
                        "Donor": np.nan,
                        "Abs Corr": np.nan,
                        "Overlap Hours": observed_count,
                    }))

            if not pred_parts:
                continue

            preds = pd.concat(pred_parts).sort_index()
            preds = preds[~preds.index.duplicated(keep="first")]
            corrected = correct_boundary_drift(preds["Filled"], target_df[param], start, end)
            preds["Filled"] = apply_physical_bounds(corrected, param)

            target_df.loc[preds.index, param] = preds["Filled"].values
            filled_count += len(preds)
            for ts, row in preds.iterrows():
                detail_rows.append({
                    "Station": station,
                    "Parameter": param,
                    "Start": start,
                    "End": end,
                    "Timestamp": ts,
                    "Filled": round(float(row["Filled"]), 6),
                    "Method": row["Method"],
                    "Donor": row["Donor"],
                    "Abs Corr": row["Abs Corr"],
                    "Overlap Hours": row["Overlap Hours"],
                })

        print(f"    filled {filled_count} hours; NaN left {int(target_df[param].isna().sum())}")

    target_df.index.name = "Date"
    output_path = OUT_DIR / f"Station{station}_filled_final.csv"
    target_df.to_csv(output_path, na_rep="NaN")
    print(f"  written: {output_path}")

    if detail_rows:
        detail_path = OUT_DIR / f"Station{station}_final_residual_fill_detail.csv"
        pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
        print(f"  written: {detail_path}")
    else:
        print("  no residual gaps filled; no detail file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill remaining residual soil NaNs after sensor-level QC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--station", type=str, nargs="*", help="Station IDs/site codes")
    parser.add_argument("--param", type=str, nargs="*", help="Parameters, e.g. SWC_5 T_20")
    parser.add_argument("--min-overlap", type=int, default=1000)
    parser.add_argument("--min-abs-corr", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    donor_pool = discover_stations()
    stations = args.station if args.station else donor_pool
    params = args.param if args.param else DEFAULT_PARAMS
    if not stations:
        print("No staged sensor-QC files found in ./output, abort.", file=sys.stderr)
        sys.exit(1)

    all_data = {station: read_station(input_path_for(station)) for station in donor_pool}
    OUT_DIR.mkdir(exist_ok=True)
    for station in stations:
        fill_station(station, params, all_data, args.min_overlap, args.min_abs_corr)


if __name__ == "__main__":
    main()

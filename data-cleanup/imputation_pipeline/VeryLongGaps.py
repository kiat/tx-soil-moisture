"""Fill >=30-day soil gaps with cross-station donor regression.

The 33-station workflow uses repaired long-gap outputs as the default input:

    output/Station{site}_filled_longgaps_repaired.csv

For each station/parameter/gap, the script chooses the donor station with the
highest absolute correlation over overlapping non-missing hours, fits a simple
linear map from donor to target, and writes predictions for the missing period.
If no regression donor is available, it falls back to the hourly mean of usable
donor stations.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from param_config import ALL_SOIL_PARAMS


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
MISS_DIR = BASE_DIR / "missing_data"
DEFAULT_PARAMS = ALL_SOIL_PARAMS


def input_path_for(station_id: str, directory=OUT_DIR) -> Path:
    candidates = [
        Path(directory) / f"Station{station_id}_filled_longgaps_repaired.csv",
        Path(directory) / f"Station{station_id}_filled_longgaps.csv",
        Path(directory) / f"Station{station_id}_filled_mediumgaps_repaired.csv",
        Path(directory) / f"Station{station_id}_filled_mediumgaps.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def load_stage_data(station_id: str, directory=OUT_DIR) -> pd.DataFrame:
    filename = input_path_for(station_id, directory)
    if not filename.exists():
        raise FileNotFoundError(f"No staged input found for Station{station_id}")
    df = pd.read_csv(filename, parse_dates=[0], index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    return ensure_hourly_regular_index(df)


def load_missing_data(station_id: str, directory=MISS_DIR) -> pd.DataFrame:
    filename = Path(directory) / f"Station{station_id}_missing_data.csv"
    return pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])


def ensure_hourly_regular_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.empty:
        return df
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(full_idx)


def filter_very_long_gaps(df_missing: pd.DataFrame, parameter: str, min_gap=720) -> pd.DataFrame:
    df_missing["Number Missing"] = pd.to_numeric(df_missing["Number Missing"], errors="coerce")
    mask = (df_missing["Parameter"] == parameter) & (df_missing["Number Missing"] >= min_gap)
    return df_missing.loc[mask].sort_values("Start Timestamp")


def discover_stations() -> List[str]:
    pat = re.compile(r"Station(.+)_filled_longgaps(?:_repaired)?\.csv")
    stations = {
        m.group(1)
        for f in OUT_DIR.glob("Station*_filled_longgaps*.csv")
        if (m := pat.match(f.name))
    }
    return sorted(stations)


def usable_series(series: pd.Series, min_std: float) -> bool:
    s = series.dropna()
    return len(s) > 0 and float(s.std()) >= min_std


def min_std_for(param: str) -> float:
    if param.startswith("SWC_"):
        return 1e-4
    if param.startswith("T_"):
        return 0.05
    return 0.0


def choose_best_donor(
    target_s: pd.Series,
    donor_dict: Dict[str, pd.Series],
    min_overlap: int,
    min_abs_corr: float,
    min_std: float,
) -> Tuple[Optional[str], float, int]:
    """Pick donor with highest |r| and enough non-missing overlap."""
    best_sid, best_r, best_overlap = None, -np.inf, 0
    if not usable_series(target_s, min_std):
        return None, float("nan"), 0

    for sid, donor_s in donor_dict.items():
        if not usable_series(donor_s, min_std):
            continue
        mask = target_s.notna() & donor_s.notna()
        overlap = int(mask.sum())
        if overlap < min_overlap:
            continue
        r = target_s[mask].corr(donor_s[mask])
        if pd.isna(r) or abs(r) < min_abs_corr:
            continue
        if abs(r) > best_r:
            best_sid, best_r, best_overlap = sid, abs(float(r)), overlap
    return best_sid, best_r, best_overlap


def fit_linear_map(y: pd.Series, x: pd.Series) -> LinearRegression:
    mask = y.notna() & x.notna()
    return LinearRegression().fit(x[mask].values.reshape(-1, 1), y[mask])


def correct_boundary_drift(preds: pd.Series, observed: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    left_ts = start - timedelta(hours=1)
    right_ts = end + timedelta(hours=1)
    has_left = left_ts in observed.index and pd.notna(observed.loc[left_ts])
    has_right = right_ts in observed.index and pd.notna(observed.loc[right_ts])

    corrected = preds.copy()
    if corrected.empty:
        return corrected
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


def donor_mean_prediction(
    idx: pd.DatetimeIndex,
    donors: Dict[str, pd.DataFrame],
    parameter: str,
    min_std: float,
) -> pd.Series:
    donor_series = {
        sid: df[parameter].reindex(idx)
        for sid, df in donors.items()
        if parameter in df.columns and usable_series(df[parameter], min_std)
    }
    if not donor_series:
        return pd.Series(dtype=float)
    combined = pd.DataFrame(donor_series)
    return combined.mean(axis=1, skipna=True).dropna()


def prediction_from_donor(
    idx: pd.DatetimeIndex,
    donor_df: pd.DataFrame,
    parameter: str,
    model: LinearRegression,
) -> pd.Series:
    x = donor_df[parameter].reindex(idx).dropna()
    if x.empty:
        return pd.Series(dtype=float)
    preds = model.predict(x.values.reshape(-1, 1))
    return pd.Series(preds, index=x.index)


def cv_metrics(target: pd.Series, donor: pd.Series, model: LinearRegression) -> Tuple[float, float]:
    mask = target.notna() & donor.notna()
    if int(mask.sum()) < 100:
        return float("nan"), float("nan")
    sample_idx = mask[mask].sample(frac=0.1, random_state=0).index
    y_true = target.loc[sample_idx]
    y_pred = model.predict(donor.loc[sample_idx].values.reshape(-1, 1))
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_squared_error(y_true, y_pred, squared=False)),
    )


def fill_station(
    station: str,
    params: Iterable[str],
    all_data: Dict[str, pd.DataFrame],
    min_overlap: int,
    min_abs_corr: float,
) -> None:
    print(f"\n=== Station {station} | very-long gap filling ===")
    df_target = all_data[station].copy()
    donors = {sid: df for sid, df in all_data.items() if sid != station}
    missing = load_missing_data(station)
    detail_rows = []

    for param in params:
        if param not in df_target.columns:
            print(f"  {param}: column missing, skip.")
            continue

        gaps = filter_very_long_gaps(missing, param)
        if gaps.empty:
            print(f"  {param}: no >=30-day gap.")
            continue

        available_donors = {sid: df for sid, df in donors.items() if param in df.columns}
        if not available_donors:
            print(f"  {param}: no donors with this parameter, skip.")
            continue

        print(f"  {param}: filling {len(gaps)} very-long gap(s)")
        min_std = min_std_for(param)
        donor_sid, corr, overlap = choose_best_donor(
            df_target[param],
            {sid: df[param] for sid, df in available_donors.items()},
            min_overlap=min_overlap,
            min_abs_corr=min_abs_corr,
            min_std=min_std,
        )

        method = "donor_mean"
        model = None
        if donor_sid is not None:
            model = fit_linear_map(df_target[param], available_donors[donor_sid][param])
            mae, rmse = cv_metrics(df_target[param], available_donors[donor_sid][param], model)
            method = "linear_donor"
            print(f"    donor={donor_sid} |r|={corr:.3f} overlap={overlap} CV_MAE={mae:.4f} CV_RMSE={rmse:.4f}")
        else:
            print("    no regression donor; using donor-mean fallback when available")

        before_count = int(df_target[param].isna().sum())
        for _, gap in gaps.iterrows():
            start, end = gap["Start Timestamp"], gap["End Timestamp"]
            idx = pd.date_range(start, end, freq="h")
            missing_idx = idx[df_target[param].reindex(idx).isna()]
            if len(missing_idx) == 0:
                continue

            pred_parts = []
            linear_preds = pd.Series(dtype=float)
            if model is not None and donor_sid is not None:
                linear_preds = prediction_from_donor(missing_idx, available_donors[donor_sid], param, model).dropna()
                if not linear_preds.empty:
                    pred_parts.append(pd.DataFrame({
                        "Filled": linear_preds,
                        "Method": "linear_donor",
                        "Donor": donor_sid,
                        "Abs Corr": corr,
                        "Overlap Hours": overlap,
                    }))

            fallback_idx = missing_idx.difference(linear_preds.index)
            if len(fallback_idx) > 0:
                mean_preds = donor_mean_prediction(fallback_idx, available_donors, param, min_std).dropna()
                if not mean_preds.empty:
                    pred_parts.append(pd.DataFrame({
                        "Filled": mean_preds,
                        "Method": "donor_mean",
                        "Donor": np.nan,
                        "Abs Corr": np.nan,
                        "Overlap Hours": np.nan,
                    }))

            if not pred_parts:
                continue

            preds = pd.concat(pred_parts).sort_index()
            preds = preds[~preds.index.duplicated(keep="first")]
            corrected = correct_boundary_drift(preds["Filled"], df_target[param], start, end)
            preds["Filled"] = apply_physical_bounds(corrected, param)

            df_target.loc[preds.index, param] = preds["Filled"].values
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

        filled_count = before_count - int(df_target[param].isna().sum())
        print(f"    filled {filled_count} hours; NaN left {int(df_target[param].isna().sum())}")

    filled_csv = OUT_DIR / f"Station{station}_filled_verylonggaps.csv"
    df_target.index.name = "Date"
    df_target.to_csv(filled_csv, na_rep="NaN")
    print(f"  written: {filled_csv}")

    if detail_rows:
        detail_csv = OUT_DIR / f"Station{station}_verylonggap_fill_detail.csv"
        pd.DataFrame(detail_rows).to_csv(detail_csv, index=False)
        print(f"  written: {detail_csv}")
    else:
        print("  no very-long gaps filled; no detail file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill >=30-day soil gaps using cross-station donor regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--station", type=str, nargs="*", help="Station IDs/site codes")
    parser.add_argument("--param", type=str, nargs="*", help="Parameters, e.g. SWC_20 T_20")
    parser.add_argument("--min-overlap", type=int, default=1000)
    parser.add_argument("--min-abs-corr", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    donor_pool = discover_stations()
    stations = args.station if args.station else donor_pool
    params = args.param if args.param else DEFAULT_PARAMS
    if not stations:
        print("No staged long-gap files found in ./output, abort.", file=sys.stderr)
        sys.exit(1)

    all_data = {station: load_stage_data(station) for station in donor_pool}
    OUT_DIR.mkdir(exist_ok=True)
    for station in stations:
        fill_station(station, params, all_data, args.min_overlap, args.min_abs_corr)


if __name__ == "__main__":
    main()

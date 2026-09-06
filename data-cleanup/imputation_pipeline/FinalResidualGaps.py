"""Fill remaining soil NaNs after validation and sensor-level QC.

This is a final non-destructive filling stage. It reads the manual-QC output
when available, otherwise the sensor-QC output:

    output/Station{site}_filled_manual_qc.csv
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
from time_index_utils import require_unique_datetime_index


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
MANUAL_QC_MASKS = BASE_DIR / "manual_qc_masks.csv"
DEFAULT_PARAMS = ALL_SOIL_PARAMS


def input_path_for(
    station: str,
    manual_stations: set[str],
    directory: Path = OUT_DIR,
) -> Path:
    suffix = "filled_manual_qc.csv" if station in manual_stations else "filled_sensor_qc.csv"
    path = Path(directory) / f"Station{station}_{suffix}"
    prerequisite = "manual-QC" if station in manual_stations else "sensor-QC"
    if not path.is_file():
        raise FileNotFoundError(
            f"FinalResidualGaps.py requires {prerequisite} input for "
            f"Station{station}: {path}"
        )
    return path


def discover_stations() -> List[str]:
    pattern = re.compile(r"Station(.+)_filled_verylonggaps_repaired\.csv")
    return sorted(
        m.group(1)
        for path in OUT_DIR.glob("Station*_filled_verylonggaps_repaired.csv")
        if (m := pattern.match(path.name))
    )


def read_station(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    return ensure_hourly_regular_index(df)


def ensure_hourly_regular_index(df: pd.DataFrame) -> pd.DataFrame:
    require_unique_datetime_index(df, "FinalResidualGaps input")
    df = df.sort_index()
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


def choose_boundary_adjusted_prediction(
    raw_preds: pd.Series,
    corrected_preds: pd.Series,
    param: str,
    min_long_run: int = 720,
) -> Tuple[pd.Series, str]:
    """Use boundary drift unless it creates a clear low-bound artifact.

    Long manually reviewed SWC gaps can have unreliable values immediately
    before or after the masked segment. In that case, forcing the fill to match
    those boundaries can push an otherwise reasonable donor prediction down to
    zero for thousands of hours. Keep the uncorrected donor prediction when that
    specific artifact is detected.
    """
    bounded_raw = apply_physical_bounds(raw_preds, param)
    bounded_corrected = apply_physical_bounds(corrected_preds, param)

    if not param.startswith("SWC_") or len(bounded_corrected) < min_long_run:
        return bounded_corrected, "applied"

    raw_near_zero = float((bounded_raw <= 0.01).mean()) if len(bounded_raw) else 0.0
    corrected_near_zero = float((bounded_corrected <= 0.01).mean()) if len(bounded_corrected) else 0.0
    raw_exact_lower = float((bounded_raw == 0.0).mean()) if len(bounded_raw) else 0.0
    corrected_exact_lower = float((bounded_corrected == 0.0).mean()) if len(bounded_corrected) else 0.0

    if (
        corrected_near_zero >= 0.50
        and raw_near_zero <= 0.20
        and corrected_exact_lower >= 0.20
        and raw_exact_lower <= 0.05
    ):
        return bounded_raw, "skipped_low_bound_artifact"

    return bounded_corrected, "applied"


def replace_swc_lower_bound_predictions(
    predictions: pd.DataFrame,
    donors: Dict[str, pd.DataFrame],
    param: str,
    min_std: float,
) -> Tuple[pd.DataFrame, int]:
    """Replace model-created exact-zero SWC fills with positive donor support."""
    if not param.startswith("SWC_") or predictions.empty:
        return predictions, 0

    result = predictions.copy()
    remaining = pd.DatetimeIndex(result.index[result["Filled"].le(0.0)])
    repaired = 0
    fallback_methods = [
        (donor_mean_prediction, "donor_mean_lower_bound_repair"),
        (donor_climatology_prediction, "donor_climatology_lower_bound_repair"),
    ]
    for predictor, method in fallback_methods:
        if not len(remaining):
            break
        fallback = predictor(remaining, donors, param, min_std)
        fallback = fallback[fallback.gt(0.0) & np.isfinite(fallback)]
        if fallback.empty:
            continue
        result.loc[fallback.index, "Filled"] = fallback
        result.loc[fallback.index, "Method"] = method
        result.loc[fallback.index, "Donor"] = np.nan
        result.loc[fallback.index, "Abs Corr"] = np.nan
        repaired += len(fallback)
        remaining = remaining.difference(fallback.index)
    return result, repaired


def prediction_frame(
    values: pd.Series,
    method: str,
    overlap_hours: int,
    donor: object = np.nan,
    abs_corr: object = np.nan,
) -> pd.DataFrame:
    """Build one consistently shaped block for the residual-fill log."""
    return pd.DataFrame(
        {
            "Filled": values,
            "Method": method,
            "Donor": donor,
            "Abs Corr": abs_corr,
            "Overlap Hours": overlap_hours,
        },
        index=values.index,
    )


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


def load_manual_masks(path: Path = MANUAL_QC_MASKS) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manual QC decision file not found: {path}")
    masks = pd.read_csv(path, parse_dates=["Start", "End"])
    required = {"Station", "Parameter", "Start", "End", "Decision"}
    missing = required - set(masks.columns)
    if missing:
        raise ValueError(f"Manual QC decision file is missing columns: {sorted(missing)}")
    masks["Station"] = masks["Station"].astype(str)
    return masks[masks["Decision"].eq("mask_and_refill")].copy()


def load_manual_refill_overrides(masks: pd.DataFrame) -> pd.DataFrame:
    if "Refill Method" not in masks.columns:
        return pd.DataFrame()
    masks = masks.copy()
    masks["Refill Method"] = masks["Refill Method"].fillna("auto").astype(str)
    return masks[masks["Refill Method"].ne("auto")].copy()


def split_run_by_refill_override(
    run: pd.DatetimeIndex,
    overrides: pd.DataFrame,
    station: str,
    param: str,
) -> List[Tuple[pd.DatetimeIndex, str]]:
    if len(run) == 0:
        return []
    methods = pd.Series("auto", index=run, dtype="object")
    if overrides.empty:
        return [(run, "auto")]

    matches = overrides[
        overrides["Station"].eq(station)
        & overrides["Parameter"].eq(param)
        & (overrides["Start"] <= run[-1])
        & (overrides["End"] >= run[0])
    ]
    for _, row in matches.iterrows():
        method = str(row["Refill Method"])
        covered = (methods.index >= row["Start"]) & (methods.index <= row["End"])
        current = methods.loc[covered]
        conflict = current.ne("auto") & current.ne(method)
        if conflict.any():
            first_conflict = current.index[conflict][0]
            raise ValueError(
                f"Conflicting manual refill methods for Station{station} {param} "
                f"at {first_conflict}: {current.loc[first_conflict]} vs {method}"
            )
        methods.loc[covered] = method

    group_ids = methods.ne(methods.shift()).cumsum()
    return [
        (pd.DatetimeIndex(group.index), str(group.iloc[0]))
        for _, group in methods.groupby(group_ids, sort=False)
    ]


def fill_station(
    station: str,
    params: Iterable[str],
    all_data: Dict[str, pd.DataFrame],
    min_overlap: int,
    min_abs_corr: float,
    refill_overrides: pd.DataFrame,
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
        segmented_runs = (
            segment
            for run in runs
            for segment in split_run_by_refill_override(run, refill_overrides, station, param)
        )
        for idx, refill_method in segmented_runs:
            start, end = idx[0], idx[-1]
            pred_parts: List[pd.DataFrame] = []
            predicted_idx = pd.DatetimeIndex([])

            if refill_method == "donor_mean":
                mean_preds = donor_mean_prediction(idx, available_donors, param, min_std).dropna()
                if not mean_preds.empty:
                    pred_parts.append(
                        prediction_frame(
                            mean_preds,
                            "donor_mean_manual_override",
                            observed_count,
                        )
                    )
                    predicted_idx = predicted_idx.union(mean_preds.index)
            elif model is not None and donor_sid is not None:
                linear_preds = linear_prediction(idx, available_donors[donor_sid][param], model).dropna()
                if not linear_preds.empty:
                    pred_parts.append(
                        prediction_frame(
                            linear_preds,
                            "linear_donor",
                            overlap,
                            donor_sid,
                            corr,
                        )
                    )
                    predicted_idx = predicted_idx.union(linear_preds.index)

            fallback_idx = idx.difference(predicted_idx)
            if len(fallback_idx) > 0:
                mean_preds = donor_mean_prediction(fallback_idx, available_donors, param, min_std).dropna()
                if not mean_preds.empty:
                    method = (
                        "donor_mean_no_target_training"
                        if observed_count < min_overlap
                        else "donor_mean_missing_linear_donor"
                    )
                    pred_parts.append(
                        prediction_frame(mean_preds, method, observed_count)
                    )
                    predicted_idx = predicted_idx.union(mean_preds.index)

            climatology_idx = idx.difference(predicted_idx)
            if len(climatology_idx) > 0:
                clim_preds = donor_climatology_prediction(climatology_idx, available_donors, param, min_std).dropna()
                if not clim_preds.empty:
                    pred_parts.append(
                        prediction_frame(
                            clim_preds,
                            "donor_climatology_no_timestamp_donor",
                            observed_count,
                        )
                    )

            if not pred_parts:
                continue

            preds = pd.concat(pred_parts).sort_index()
            require_unique_datetime_index(preds, "FinalResidualGaps predictions")
            if refill_method == "donor_mean":
                preds["Filled"] = apply_physical_bounds(preds["Filled"], param)
                boundary_adjustment = "skipped_manual_donor_mean_override"
            else:
                corrected = correct_boundary_drift(preds["Filled"], target_df[param], start, end)
                preds["Filled"], boundary_adjustment = choose_boundary_adjusted_prediction(
                    preds["Filled"],
                    corrected,
                    param,
                )
            preds, lower_bound_repairs = replace_swc_lower_bound_predictions(
                preds,
                available_donors,
                param,
                min_std,
            )
            if lower_bound_repairs:
                boundary_adjustment += (
                    f"; positive_donor_lower_bound_repairs={lower_bound_repairs}"
                )

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
                    "Boundary Adjustment": boundary_adjustment,
                    "Refill Override": refill_method,
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
        print("No validated very-long-gap station cohort found in ./output, abort.", file=sys.stderr)
        sys.exit(1)

    manual_masks = load_manual_masks()
    manual_stations = set(manual_masks["Station"])
    load_stations = sorted(set(donor_pool) | set(stations))
    all_data = {
        station: read_station(input_path_for(station, manual_stations))
        for station in load_stations
    }
    refill_overrides = load_manual_refill_overrides(manual_masks)
    OUT_DIR.mkdir(exist_ok=True)
    for station in stations:
        fill_station(station, params, all_data, args.min_overlap, args.min_abs_corr, refill_overrides)


if __name__ == "__main__":
    main()

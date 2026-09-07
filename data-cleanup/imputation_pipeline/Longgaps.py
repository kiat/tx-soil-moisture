"""Fill 168- to 719-hour soil gaps with XGBoost.

The 33-station workflow requires validated medium-gap outputs:

    output/Station{site}_filled_mediumgaps_repaired.csv

Run validate_mediumgaps.py with --write-repaired before this stage. Station IDs
are treated as strings, so both site codes (CB01) and old numeric IDs can be
used.
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from param_config import ALL_SOIL_PARAMS
from time_index_utils import require_unique_datetime_index
from soil_source_coverage import load_soil_coverage

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent
OUT_DIR   = BASE_DIR / "output"
MISS_DIR  = BASE_DIR / "missing_data"
DEFAULT_PARAMS = ALL_SOIL_PARAMS


def input_path_for(station_id, directory=OUT_DIR):
    path = Path(directory) / f"Station{station_id}_filled_mediumgaps_repaired.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Longgaps.py requires validated medium-gap input for Station{station_id}: "
            f"{path}. Run validate_mediumgaps.py --write-repaired first."
        )
    return path


def load_medium_data(station_id, directory=OUT_DIR):
    filename = input_path_for(station_id, directory)
    df = pd.read_csv(filename, parse_dates=[0], index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df = ensure_hourly_regular_index(df)
    return df

def load_missing_data(station_id, directory=MISS_DIR, coverage=None):
    filename = Path(directory) / f"Station{station_id}_missing_data.csv"
    df = pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])
    return (coverage or load_soil_coverage(station_id)).restrict_gaps(df)


def ensure_hourly_regular_index(df: pd.DataFrame) -> pd.DataFrame:
    require_unique_datetime_index(df, "Longgaps input")
    df = df.sort_index()
    if df.empty:
        return df
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(full_idx)


def ensure_driver_columns(df: pd.DataFrame) -> dict[str, str]:
    """Create model-only drivers while preserving unavailable values as NaN."""
    sources = {}
    if "Ppt" in df.columns:
        df["Ppt_model"] = pd.to_numeric(df["Ppt"], errors="coerce")
        sources["Ppt"] = "Ppt"
    else:
        df["Ppt_model"] = pd.Series(np.nan, index=df.index)
        sources["Ppt"] = "unavailable"

    if "Tair" in df.columns and df["Tair"].notna().any():
        tair = pd.to_numeric(df["Tair"], errors="coerce")
        sources["Tair"] = "Tair"
    else:
        temp_cols = [c for c in ["T_5", "T_10", "T_20", "T_50"] if c in df.columns]
        if temp_cols:
            tair = df[temp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            sources["Tair"] = "soil_temperature_proxy"
        else:
            tair = pd.Series(np.nan, index=df.index)
            sources["Tair"] = "unavailable"
    df["Tair_model"] = tair

    if "Srad" in df.columns:
        df["Srad_model"] = pd.to_numeric(df["Srad"], errors="coerce")
        sources["Srad"] = "Srad" if df["Srad_model"].notna().any() else "unavailable"
    else:
        df["Srad_model"] = pd.Series(np.nan, index=df.index)
        sources["Srad"] = "unavailable"
    df.attrs["soil_driver_sources"] = sources
    return sources


def _complete_sum(values: pd.Series):
    if values.empty or values.isna().any():
        return np.nan
    return values.sum()


def _complete_mean(values: pd.Series):
    if values.empty or values.isna().any():
        return np.nan
    return values.mean()


def driver_window_missing_counts(df, index, window=168):
    """Count feature rows whose environmental lookback contains missing data."""
    counts = {}
    for driver in ["Ppt", "Tair", "Srad"]:
        column = f"{driver}_model"
        values = (
            df[column]
            if column in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        prior_missing = values.isna().shift(1).rolling(
            window, min_periods=1
        ).max().fillna(driver == "Ppt").astype(bool)
        counts[driver] = int(
            prior_missing.reindex(index, fill_value=True).sum()
        )
    return counts

def filter_long_gaps(df_missing, parameter, min_gap=168, max_gap=720):
    df_missing["Number Missing"] = pd.to_numeric(df_missing["Number Missing"], errors="coerce")
    mask = (
        (df_missing["Parameter"] == parameter)
        & (df_missing["Number Missing"] >= min_gap)
        & (df_missing["Number Missing"] < max_gap)
    )
    return df_missing.loc[mask].sort_values("Start Timestamp")


def make_features(df, ts, param, window=168):
    hist = df.loc[ts - timedelta(hours=window) : ts - timedelta(hours=1)]
    target_hist = hist[param] if param in hist.columns else pd.Series(dtype=float)
    ppt = hist["Ppt_model"] if "Ppt_model" in hist.columns else pd.Series(dtype=float)
    tair = hist["Tair_model"] if "Tair_model" in hist.columns else pd.Series(dtype=float)
    srad = hist["Srad_model"] if "Srad_model" in hist.columns else pd.Series(dtype=float)
    ppt_last6h = _complete_sum(ppt.tail(6))

    feat = {
        "last": target_hist.ffill().iloc[-1] if target_hist.notna().any() else np.nan,
        "mean": target_hist.mean(),
        "std": target_hist.std(),
        "min": target_hist.min(),
        "max": target_hist.max(),
        "ppt_sum7d":  _complete_sum(ppt),
        "ppt_sum24h": _complete_sum(ppt.tail(24)),
        "ppt_last3h": _complete_sum(ppt.tail(3)),
        "ppt_flag": (
            int(ppt_last6h > 0) if pd.notna(ppt_last6h) else np.nan
        ),
        "temp_mean": _complete_mean(tair),
        "temp_last": tair.iloc[-1] if len(tair) else np.nan,
        "srad_mean": _complete_mean(srad),
        "doy": ts.dayofyear,
        "hour": ts.hour,
        "sin_hour": np.sin(2 * np.pi * ts.hour / 24),
        "cos_hour": np.cos(2 * np.pi * ts.hour / 24),
        "sin_doy": np.sin(2 * np.pi * ts.dayofyear / 366),
        "cos_doy": np.cos(2 * np.pi * ts.dayofyear / 366),
    }
    return pd.Series(feat)

# ────────────────────────────────────────────────────────────
#  XGB & rolling fill
# ────────────────────────────────────────────────────────────
def train_xgb(df, param, min_train=168):
    idx = df[param].dropna().index
    if len(idx) < min_train:
        raise ValueError(f"Only {len(idx)} observed values available for {param}")
    X = pd.DataFrame([make_features(df, t, param) for t in idx])
    y = df.loc[idx, param]

    xgb = XGBRegressor(
        n_estimators   = 250,
        learning_rate  = 0.05,
        max_depth      = 4,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        objective      = "reg:squarederror",
        n_jobs         = -1,
        random_state   = 42,
        tree_method    = "hist"         
    )
    xgb.fit(X, y)
    return xgb

def rolling_fill(model, df, idx, param):
    preds = []
    for ts in idx:
        x_row = make_features(df, ts, param).to_frame().T
        y_hat = model.predict(x_row)[0]
        preds.append(y_hat)
        df.at[ts, param] = y_hat
    return pd.Series(preds, index=idx)


def correct_boundary_drift(preds, observed, start, end):
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
            drift = np.linspace(left_delta, right_delta, len(corrected))
            corrected = corrected + drift
    elif has_left:
        corrected.iloc[0] = 0.5 * (observed.loc[left_ts] + corrected.iloc[0])
    elif has_right:
        corrected.iloc[-1] = 0.5 * (observed.loc[right_ts] + corrected.iloc[-1])
    return corrected


def apply_physical_bounds(values, param):
    if param.startswith("SWC_"):
        return values.clip(lower=0.0, upper=0.6)
    if param.startswith("T_"):
        return values.clip(lower=-30.0, upper=60.0)
    return values


def fill_long_gaps_xgb_drift(df, gaps, param, station_id, coverage=None):
    coverage = coverage or load_soil_coverage(station_id)
    for _, gap in gaps.iterrows():
        coverage.require_interval(gap["Start Timestamp"], gap["End Timestamp"], param)
    training_missing = driver_window_missing_counts(
        df, df[param].dropna().index
    )
    model = train_xgb(df.copy(), param)
    work = df.copy()
    filled = work[param].copy()
    log = []

    for _, g in gaps.iterrows():
        start = g["Start Timestamp"]
        end = g["End Timestamp"]
        idx = pd.date_range(start, end, freq="h")
        prediction_missing = driver_window_missing_counts(df, idx)
        handling = (
            "xgboost_native_missing"
            if any(training_missing.values()) or any(prediction_missing.values())
            else "complete_environmental_drivers"
        )
        driver_sources = df.attrs.get("soil_driver_sources", {})
        if handling == "xgboost_native_missing":
            print(
                "  -> driver fallback: XGBoost native missing-value routing "
                f"for {start}-{end}"
            )
        preds = rolling_fill(model, work, idx, param)
        preds = correct_boundary_drift(preds, filled, start, end)
        preds = apply_physical_bounds(preds, param)

        filled.loc[idx] = preds
        work.loc[idx, param] = preds
        for ts, val in preds.items():
            log.append({
                "Station":   station_id,
                "Parameter": param,
                "Start":     start,
                "End":       end,
                "Timestamp": ts,
                "Filled":    val,
                "Driver Handling": handling,
                "Driver Sources": json.dumps(driver_sources, sort_keys=True),
                "Training Feature Rows With Missing Drivers": json.dumps(
                    training_missing, sort_keys=True
                ),
                "Prediction Feature Rows With Missing Drivers": json.dumps(
                    prediction_missing, sort_keys=True
                ),
            })
    return filled, pd.DataFrame(log)


# ────────────────────────────────────────────────────────────
#  Driver per station
# ────────────────────────────────────────────────────────────
def process_station(station, params):
    print(f"\n=== Station {station} ===")

    df = load_medium_data(station)
    coverage = load_soil_coverage(station)
    coverage.assert_frame(df)
    ensure_driver_columns(df)
    miss_tbl = load_missing_data(station, coverage=coverage)

    log_all = []
    for p in params:
        if p not in df.columns:
            print(f"  {p}: column missing, skip.")
            continue
        gaps = filter_long_gaps(miss_tbl, p)
        if gaps.empty:
            print(f"  {p}: no 7–30 day gap.")
            continue

        print(f"  {p}: filling {len(gaps)} long gap(s)…")
        try:
            filled, log = fill_long_gaps_xgb_drift(df.copy(), gaps, p, station_id=station, coverage=coverage)
        except Exception as exc:
            print(f"  {p}: skip long gaps ({exc})")
            continue
        df[p] = filled
        if not log.empty:
            log_all.append(log)

    # write results
    out_clean = OUT_DIR / f"Station{station}_filled_longgaps.csv"
    output_df = df.drop(columns=["Ppt_model", "Tair_model", "Srad_model"], errors="ignore")
    coverage.assert_frame(output_df)
    output_df.to_csv(out_clean)
    print("  • written:", out_clean)

    if log_all:
        out_detail = OUT_DIR / f"Station{station}_longgap_fill_detail.csv"
        pd.concat(log_all, ignore_index=True).to_csv(out_detail, index=False)
        print("  • written:", out_detail)


# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description="Fill 7–30 day gaps with XGBoost (SWC & soil temperature).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--station", type=str, nargs="*", help="Station IDs/site codes")
    ap.add_argument("--param", type=str, nargs="*", help="Parameters, e.g. SWC_20 T_20")
    return ap.parse_args()


def discover_stations():
    pat = re.compile(r"Station(.+)_filled_mediumgaps_repaired\.csv")
    stations = {
        m.group(1)
        for f in OUT_DIR.glob("Station*_filled_mediumgaps_repaired.csv")
        if (m := pat.match(f.name))
    }
    return sorted(stations)


def main():
    args = parse_args()
    stations = args.station if args.station else discover_stations()
    params = args.param if args.param else DEFAULT_PARAMS

    if not stations:
        print("No station files found in ./output, abort.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    for st in stations:
        process_station(st, params)

if __name__ == "__main__":
    main()

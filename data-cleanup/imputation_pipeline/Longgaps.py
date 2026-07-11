"""Fill 7- to 30-day soil gaps with XGBoost.

The 33-station workflow uses repaired medium-gap outputs as the default input:

    output/Station{site}_filled_mediumgaps_repaired.csv

If a repaired file does not exist, the script falls back to the raw medium-gap
output. Station IDs are treated as strings, so both site codes (CB01) and old
numeric IDs can be used.
"""
import argparse
import re
import sys
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from param_config import ALL_SOIL_PARAMS

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent
OUT_DIR   = BASE_DIR / "output"
MISS_DIR  = BASE_DIR / "missing_data"
DEFAULT_PARAMS = ALL_SOIL_PARAMS


def input_path_for(station_id, directory=OUT_DIR):
    candidates = [
        Path(directory) / f"Station{station_id}_filled_mediumgaps_repaired.csv",
        Path(directory) / f"Station{station_id}_filled_mediumgaps.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def load_medium_data(station_id, directory=OUT_DIR):
    filename = input_path_for(station_id, directory)
    if not filename.exists():
        raise FileNotFoundError(f"No medium-gap input found for Station{station_id}")
    df = pd.read_csv(filename, parse_dates=[0], index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df = ensure_hourly_regular_index(df)
    return df

def load_missing_data(station_id, directory=MISS_DIR):
    filename = Path(directory) / f"Station{station_id}_missing_data.csv"
    df = pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])
    return df


def ensure_hourly_regular_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.empty:
        return df
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(full_idx)


def ensure_driver_columns(df: pd.DataFrame) -> None:
    """Create local driver columns used by feature engineering."""
    if "Ppt" in df.columns:
        df["Ppt"] = df["Ppt"].reindex(df.index).fillna(0.0)
    else:
        df["Ppt"] = 0.0

    if "Tair" in df.columns and df["Tair"].notna().any():
        tair = df["Tair"]
    else:
        temp_cols = [c for c in ["T_5", "T_10", "T_20", "T_50"] if c in df.columns]
        if temp_cols:
            tair = df[temp_cols].mean(axis=1)
        else:
            tair = pd.Series(np.nan, index=df.index)
    df["Tair_model"] = tair.ffill().bfill().fillna(0.0)

    if "Srad" in df.columns and df["Srad"].notna().any():
        df["Srad_model"] = df["Srad"].fillna(0.0)
    else:
        df["Srad_model"] = 0.0

def filter_long_gaps(df_missing, parameter, min_gap=168, max_gap=720):
    df_missing["Number Missing"] = pd.to_numeric(df_missing["Number Missing"], errors="coerce")
    mask = (
        (df_missing["Parameter"] == parameter)
        & (df_missing["Number Missing"] >= min_gap)
        & (df_missing["Number Missing"] <= max_gap)
    )
    return df_missing.loc[mask].sort_values("Start Timestamp")


def make_features(df, ts, param, window=168):
    hist = df.loc[ts - timedelta(hours=window) : ts - timedelta(hours=1)]
    target_hist = hist[param] if param in hist.columns else pd.Series(dtype=float)
    ppt = hist["Ppt"] if "Ppt" in hist.columns else pd.Series(dtype=float)
    tair = hist["Tair_model"] if "Tair_model" in hist.columns else pd.Series(dtype=float)
    srad = hist["Srad_model"] if "Srad_model" in hist.columns else pd.Series(dtype=float)

    feat = {
        "last": target_hist.ffill().iloc[-1] if target_hist.notna().any() else np.nan,
        "mean": target_hist.mean(),
        "std": target_hist.std(),
        "min": target_hist.min(),
        "max": target_hist.max(),
        "ppt_sum7d":  ppt.sum(),
        "ppt_sum24h": ppt.tail(24).sum(),
        "ppt_last3h": ppt.tail(3).sum(),
        "ppt_flag": int(ppt.tail(6).sum() > 0) if len(ppt) else 0,
        "temp_mean": tair.mean(),
        "temp_last": tair.ffill().iloc[-1] if tair.notna().any() else np.nan,
        "srad_mean": srad.mean(),
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


def fill_long_gaps_xgb_drift(df, gaps, param, station_id):
    model = train_xgb(df.copy(), param)
    work = df.copy()
    filled = work[param].copy()
    log = []

    for _, g in gaps.iterrows():
        start = g["Start Timestamp"]
        end = g["End Timestamp"]
        idx = pd.date_range(start, end, freq="h")
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
                "Filled":    val
            })
    return filled, pd.DataFrame(log)


# ────────────────────────────────────────────────────────────
#  Driver per station
# ────────────────────────────────────────────────────────────
def process_station(station, params, ref_station = 3):
    print(f"\n=== Station {station} ===")

    df = load_medium_data(station)
    ensure_driver_columns(df)
    miss_tbl = load_missing_data(station)

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
            filled, log = fill_long_gaps_xgb_drift(df.copy(), gaps, p, station_id=station)
        except Exception as exc:
            print(f"  {p}: skip long gaps ({exc})")
            continue
        df[p] = filled
        if not log.empty:
            log_all.append(log)

    # write results
    out_clean = OUT_DIR / f"Station{station}_filled_longgaps.csv"
    df.to_csv(out_clean)
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
    pat = re.compile(r"Station(.+)_filled_mediumgaps(?:_repaired)?\.csv")
    stations = {
        m.group(1)
        for f in OUT_DIR.glob("Station*_filled_mediumgaps*.csv")
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

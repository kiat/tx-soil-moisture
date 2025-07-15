# -----------------------------------------------------------
#   Longgaps.py  –  fill every 7-30-day gap via XGBoost 
#
#   python Longgaps.py                     #   all stations, all SWC
#   python Longgaps.py --station 2         #   only Station 2
#   python Longgaps.py --param SWC_20      #   all stations, that param
#   python Longgaps.py --station 1 --param SWC_50
#
#   Outputs (per station):
#       output/StationX_filled_longgaps.csv      – full series after filling
#       output/StationX_longgap_fill_detail.csv  – table of every gap filled
# -----------------------------------------------------------


# Import libraries
import argparse, sys
import warnings
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

OUT_DIR   = Path("output")

def load_cleaned_data(station_id, directory = "./output"):
    filename = Path(directory) / f"Station{station_id}_filled_mediumgaps.csv"
    df = pd.read_csv(filename, parse_dates=[0], index_col=0)
    df.index = pd.DatetimeIndex(df.index, freq="H")
    return df

def load_missing_data(station_id, directory="../EDA_2025/missing_data"):
    filename = f"{directory}/Station{station_id}_missing_data.csv"
    df = pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])
    return df

def filter_long_gaps(df_missing, parameter, min_gap=168, max_gap=720):
    df_missing["Number Missing"] = pd.to_numeric(df_missing["Number Missing"], errors="coerce")
    mask = (
        (df_missing["Parameter"] == parameter)
        & (df_missing["Number Missing"] >= min_gap)
        & (df_missing["Number Missing"] <= max_gap)
    )
    return df_missing.loc[mask].sort_values("Start Timestamp")

def baseline_predict(series, gap_slice):
    # seasonal mean  
    seasonal_key   = series.index.strftime("%m-%d-%H")
    seasonal_means = series.groupby(seasonal_key).transform("mean")

    # global linear trend
    mask_not_na = series.notna()
    x_train = series.index[mask_not_na].view("int64").reshape(-1, 1)
    y_train = series[mask_not_na].to_numpy()
    trend_model = LinearRegression().fit(x_train, y_train)

    # trend prediction for the full index
    x_full   = series.index.view("int64").reshape(-1, 1)
    trend    = trend_model.predict(x_full)

    # combine & return slice
    baseline = seasonal_means + (trend - trend.mean())
    return baseline.loc[gap_slice]

def make_features(df, ts, param, window=168):
    hist = df.loc[ts - timedelta(hours=window) : ts - timedelta(hours=1)]
    feat = {
        "last": hist[param].iloc[-1] if not hist.empty else np.nan,
        "mean": hist[param].mean(),
        "std": hist[param].std(),
        "ppt_sum7d":  hist["Ppt"].sum(),                          # 168h
        "ppt_sum24h": hist["Ppt"].tail(24).sum(),                 # 24h
        "ppt_last3h": hist["Ppt"].tail(3).sum(),                  # 3h
        "ppt_flag":   int(hist["Ppt"].tail(6).sum() > 0),         # check if rain within 6h
        "temp_mean": hist["Tair"].mean(),
        "doy": ts.dayofyear,
        "hour": ts.hour,
    }
    return pd.Series(feat)

# ────────────────────────────────────────────────────────────
#  XGB & rolling fill
# ────────────────────────────────────────────────────────────
def train_xgb(df, param):
    idx = df[param].dropna().index
    X   = pd.DataFrame([make_features(df, t, param) for t in idx])
    y   = df.loc[idx, param]

    xgb = XGBRegressor(
        n_estimators   = 300,
        learning_rate  = 0.05,
        max_depth      = 6,
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

def fill_long_gaps_xgb_drift(df, gaps, param, station_id):
    model  = train_xgb(df.copy(), param)
    filled = df[param].copy()
    log    = []

    for _, g in gaps.iterrows():
        idx   = pd.date_range(g["Start Timestamp"], g["End Timestamp"], freq="H")
        preds = rolling_fill(model, df.copy(), idx, param)

        # drift correction
        before_ts, after_ts = g["Start Timestamp"]-timedelta(hours=1), g["End Timestamp"]+timedelta(hours=1)
        v0, v1 = df[param].get(before_ts, np.nan), df[param].get(after_ts, np.nan)
        if pd.notna(v0) and pd.notna(v1):
            drift = np.linspace(v0-preds.iloc[0], v1-preds.iloc[-1], len(preds))
            preds = preds + drift

        filled.loc[idx] = preds
        for ts, val in preds.items():
            log.append({
                "Station":   station_id,
                "Parameter": param,
                "Start":     before_ts,
                "End":       after_ts,
                "Timestamp": ts,
                "Filled":    val
            })
    return filled, pd.DataFrame(log)


# ────────────────────────────────────────────────────────────
#  Driver per station
# ────────────────────────────────────────────────────────────
def process_station(station, params, ref_station = 3):
    print(f"\n=== Station {station} ===")

    df   = load_cleaned_data(station)
    df_ref = load_cleaned_data(ref_station)
    df["Ppt"] = df["Ppt"].fillna(df_ref["Ppt"]).fillna(0)

    log_all = []
    for p in params:
        gaps = filter_long_gaps(load_missing_data(station), p)
        if gaps.empty:
            print(f"  {p}: no 7–30 day gap.")
            continue

        print(f"  {p}: filling {len(gaps)} long gap(s)…")
        filled, log = fill_long_gaps_xgb_drift(df.copy(), gaps, p, station_id=station)
        df[p] = filled                       
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
        description="Fill 7–30 day gaps with XGBoost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--station", type=int, help="Station number (int)")
    ap.add_argument("--param",   type=str, help="Parameter, e.g. SWC_20")
    return ap.parse_args()

def main():
    args = parse_args()
    stations = ([args.station] if args.station is not None
                else sorted(int(f.stem.replace("Station","").split("_")[0])
                            for f in OUT_DIR.glob("Station*_filled_mediumgaps.csv")))
    params = ([args.param] if args.param is not None else ["SWC_5", "SWC_10", "SWC_20", "SWC_50"])

    if not stations:
        print("No station files found in ./output, abort.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    for st in stations:
        process_station(st, params)

if __name__ == "__main__":
    main()
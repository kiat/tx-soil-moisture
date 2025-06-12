# -----------------------------------------------------------
# Mediumgaps.py  -  Fill 24‑ to 168‑hour gaps via SARIMAX
# -----------------------------------------------------------
# usage examples
#    python Mediumgaps.py                       # all stations & all SWC columns
#    python Mediumgaps.py --station 2           # only station 2
#    python Mediumgaps.py --param SWC_20        # all stations, only SWC_20
#    python Mediumgaps.py --station 3 --param SWC_50  # specific combo
#
# Output files (per station)
#    output/StationX_filled_mediumgaps.csv       - cleaned series after filling
#    output/StationX_mediumgap_fill_detail.csv   - long‑form log of every value written
# -----------------------------------------------------------

# Import libraries
import warnings
import argparse, re
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

# Paths to data directories
CLEAN_DIR = Path("./output")                 # short‑gap‑filled inputs
MISS_DIR  = Path("../EDA_2025/missing_data")
OUT_DIR   = Path("./output")



def load_cleaned_data(station_id):
    filename = CLEAN_DIR / f"Station{station_id}_filled_shortgaps.csv"
    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df.index.freq = 'H'
    return df

def load_missing_data(station_id):
    filename = MISS_DIR / f"Station{station_id}_missing_data.csv"
    df = pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])
    return df



# Get medium gaps
def filter_medium_gaps(df_missing, parameter="SWC_5", min_gap=24, max_gap=168):
    # Convert "Number Missing" column to numeric
    df_missing["Number Missing"] = pd.to_numeric(df_missing["Number Missing"], errors="coerce")
    mask = (df_missing["Parameter"] == parameter) & \
           (df_missing["Number Missing"] >= min_gap) & (df_missing["Number Missing"] <= max_gap)
    return df_missing.loc[mask].sort_values("Start Timestamp")



# Fit SARIMAX model and predict missing values
def sarima_forecast(y, s_ts, e_ts, exog, ctx_days = 7, max_pq = 3, max_PQ = 2):
    # Determine the training window up to one hour before gap
    train_start = s_ts - timedelta(days=ctx_days)
    train_end = s_ts - timedelta(hours=1)

    # Extract and clean training data
    y_window = y.loc[train_start:train_end]
    y_train = y_window.dropna()
    if y_train.empty:
        print(f"[skip] No sufficient training data {train_start}–{train_end}")
        return None, None
    y_train.index = pd.DatetimeIndex(y_train.index, freq="H")

    # Prepare exogenous data
    X_train = X_pred = None
    if exog is not None:
        exog_window = exog.loc[train_start:train_end]
        X_train = exog_window.reindex(y_train.index).fillna(0)
        pred_index = pd.date_range(s_ts, e_ts, freq="H")
        X_pred = exog.reindex(pred_index).fillna(0)

    # Automatic model order selection with daily seasonality
    auto = auto_arima(
        y_train,
        exogenous=X_train,
        seasonal=True, m=24,
        d=None, D=1,
        start_p=1, start_q=1, max_p=max_pq, max_q=max_pq,
        start_P=0, start_Q=0, max_P=max_PQ, max_Q=max_PQ,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )
    p, d, q = auto.order
    P, D, Q, s = auto.seasonal_order
    print(f"  → SARIMA({p},{d},{q})x({P},{D},{Q},{s})24h")

    # Fit the SARIMAX model with frequency parameter
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq='H'
    )
    res = model.fit(method='powell', maxiter=300, disp=False)

    # Check residuals for autocorrelation
    lb_p = acorr_ljungbox(res.resid, lags=[24], return_df=True)["lb_pvalue"].iat[0]
    if lb_p < 0.05 and (max_pq < 5 or max_PQ < 3):
        # Retry with larger parameter bounds
        return sarima_forecast(
            y, s_ts, e_ts, exog,
            ctx_days=ctx_days,
            max_pq=max_pq+1,
            max_PQ=max_PQ+1
        )

    # Forecast the gap interval
    forecast_index = pd.date_range(s_ts, e_ts, freq="H")
    fc = res.forecast(steps=len(forecast_index), exog=X_pred)
    fc.index = forecast_index
    return fc, res



# Iterate through each medium gap, fit SARIMAX, and write predictions back
def fill_medium_gaps(series, gaps, exog, gap_log, station, param, ctx_days=7):
    filled = series.copy()
    for _, row in gaps.iterrows():
        s_ts = row["Start Timestamp"]
        e_ts = row["End Timestamp"]
        idx = pd.date_range(s_ts, e_ts, freq="H")

        fc, _ = sarima_forecast(filled, s_ts, e_ts, exog, ctx_days)
        if fc is None:
            continue
        # Smooth edges between original and predicted values
        if (s_ts - timedelta(hours=1)) in filled.index:
            left_val = filled.loc[s_ts - timedelta(hours=1)] 
            fc.iloc[0] = 0.5 * (left_val + fc.iloc[0])
        if (e_ts + timedelta(hours=1)) in filled.index:
            right_val = filled.loc[e_ts + timedelta(hours=1)]
            fc.iloc[-1] = 0.5 * (right_val + fc.iloc[-1])

        filled.loc[idx] = fc
        gap_log.extend({
            "Station": station, "Parameter": param,
            "Start": s_ts, "End": e_ts,
            "Timestamp": t, "Filled": v
        } for t, v in fc.items())
    return filled



# Fill medium gaps for each SWC parameter and save outputs
def process_station(station, params):
    df = load_cleaned_data(station)
    miss_tbl = load_missing_data(station)
    df_ref = load_cleaned_data(3)  # Ppt fallback

    log = []
    any_fill = False

    for p in params:
        mgaps = filter_medium_gaps(miss_tbl, p)
        if mgaps.empty:
            print(f"  {p}: no 24–168 h gaps")
            continue
        print(f"  {p}: filling {len(mgaps)} gaps")

        exog = df["Ppt"].fillna(df_ref["Ppt"]).fillna(0)
        df[p] = fill_medium_gaps(df[p], mgaps, exog, log, station, p)
        any_fill = True

    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / f"Station{station}_filled_mediumgaps.csv")
    if log:
        pd.DataFrame(log).to_csv(
            OUT_DIR / f"Station{station}_mediumgap_fill_detail.csv", index=False)
    status = " (unchanged)" if not any_fill else ""
    print(f"→ saved Station{station}_filled_mediumgaps.csv{status}\n")


# --------------- CLI helpers ---------------------------

def discover_stations():
    pat = re.compile(r"Station(\d+)_filled_shortgaps\.csv")
    return sorted(int(m.group(1)) for fn in CLEAN_DIR.glob("Station*_filled_shortgaps.csv")
                  if (m := pat.match(fn.name)))

def parse_args():
    p = argparse.ArgumentParser("Fill 24–168 h gaps via SARIMAX")
    p.add_argument("--station", type=int, nargs="*", help="station IDs")
    p.add_argument("--param",   type=str, nargs="*", help="SWC columns")
    return p.parse_args()

# --------------- main entry ----------------------------

def main():
    args = parse_args()
    stations = args.station if args.station else discover_stations()
    params   = args.param if args.param else ["SWC_5", "SWC_10", "SWC_20", "SWC_50"]

    print("Stations :", stations)
    print("Parameters:", params, "\n")

    for sid in stations:
        print(f"=== Station {sid} ===")
        process_station(sid, params)

if __name__ == "__main__":
    main()

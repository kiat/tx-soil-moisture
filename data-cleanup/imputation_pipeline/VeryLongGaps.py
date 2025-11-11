# -----------------------------------------------------------
#   VeryLongGaps.py  –  fill every ≥30-day gap via
#   cross-station linear regression.
#
#   Examples:
#     python VeryLongGaps.py                         # all stations, all SWC
#     python VeryLongGaps.py --station 2             # only Station 2
#     python VeryLongGaps.py --param SWC_20          # all stations, that param
#     python VeryLongGaps.py --station 1 --param SWC_50
#
#   Outputs (per station):
#     output/StationX_filled_verylonggaps.csv            – full series after filling
#     output/StationX_verylonggap_fill_detail.csv        – table of each gap filled
# -----------------------------------------------------------

import argparse, sys, warnings
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
BASE_DIR  = Path(__file__).resolve().parent    
CLEAN_DIR = BASE_DIR / "output"
MISS_DIR  = BASE_DIR / "missing_data"
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

def load_cleaned_data(station_id, directory=CLEAN_DIR):
    filename = Path(directory) / f"Station{station_id}_filled_longgaps.csv"
    df = pd.read_csv(filename, parse_dates=[0], index_col=0)
    df.index = pd.DatetimeIndex(df.index, freq="H")
    return df

def load_missing_data(station_id, directory=MISS_DIR):
    filename = Path(directory) / f"Station{station_id}_missing_data.csv"
    df = pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])
    return df

def filter_very_long_gaps(df_missing, parameter, min_gap=720):
    df_missing["Number Missing"] = pd.to_numeric(
        df_missing["Number Missing"], errors="coerce"
    )
    mask = (df_missing["Parameter"] == parameter) & (
        df_missing["Number Missing"] >= min_gap
    )
    return df_missing.loc[mask].sort_values("Start Timestamp")

def choose_best_donor(target_s, donor_dict):
    """Pick donor with highest |r| having ≥ 1000h non-NaN overlap."""
    best_sid, best_r = None, -np.inf
    for sid, s in donor_dict.items():
        mask = target_s.notna() & s.notna()
        if mask.sum() < 1000: # At least 6 weeks here
            continue
        r = target_s[mask].corr(s[mask])
        if abs(r) > best_r:
            best_sid, best_r = sid, abs(r)
    return best_sid, best_r

def fit_linear_map(y, x):
    """Fit y ≈ a·x + b on overlapping hours."""
    m = y.notna() & x.notna()
    mdl = LinearRegression().fit(x[m].values.reshape(-1,1), y[m])
    return mdl

# ---------- main fill routine -------------------------------------------
def fill_station(sta, params, donor_ids, win_days=14):
    print(f"\n=== Station {sta} | very-long gap filling ===")
    df_before = load_cleaned_data(sta)
    df_tgt    = df_before.copy()
    donors    = {sid: load_cleaned_data(sid) for sid in donor_ids if sid != sta}
    missing   = load_missing_data(sta)
    detail_rows = []

    for p in params:
        print(f"\n[Param] {p}")
        gaps = filter_very_long_gaps(missing, p)
        if gaps.empty:
            print("  • No ≥30-day gaps – skip"); continue

        donor_sid, r = choose_best_donor(df_tgt[p],
                                         {sid: donors[sid][p] for sid in donors})
        if donor_sid is None:
            print("  • No donor with ≥500 h overlap – skip"); continue
        print(f"  • donor S{donor_sid}  |r|={r:.3f}")

        model = fit_linear_map(df_tgt[p], donors[donor_sid][p])
        a, b  = model.coef_[0], model.intercept_
        print(f"    y ≈ {a:.3f}·x + {b:.3f}")

        for _, g in gaps.iterrows():
            start, end = g["Start Timestamp"], g["End Timestamp"]
            idx = pd.date_range(start, end, freq="H")
            x   = donors[donor_sid].loc[idx, p].dropna()
            if x.empty:  # donor also missing
                continue
            preds = model.predict(x.values.reshape(-1, 1))
            df_tgt.loc[x.index, p] = preds

            for ts, val in zip(x.index, preds):
                detail_rows.append({
                    "Station"   : sta,
                    "Parameter" : p,
                    "Start"     : start,
                    "End"       : end,
                    "Timestamp" : ts,
                    "Filled"    : round(float(val), 6) 
                })

        nan_left = df_tgt[p].isna().sum()
        print(f"    gaps filled, NaN left {nan_left}")

        # quick 10 % CV
        joint = df_tgt[p].notna() & donors[donor_sid][p].notna()
        if joint.sum() >= 100:
            idx = joint[joint].sample(frac=0.1, random_state=0).index
            mae = mean_absolute_error(df_tgt.loc[idx,p],
                                      model.predict(donors[donor_sid]
                                                    .loc[idx,p].values.reshape(-1,1)))
            rmse = mean_squared_error(df_tgt.loc[idx,p],
                                      model.predict(donors[donor_sid]
                                                    .loc[idx,p].values.reshape(-1,1)),
                                      squared=False)
            print(f"    CV  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # save filled series
    filled_csv = OUT_DIR / f"Station{sta}_filled_verylonggaps.csv"
    df_tgt.to_csv(filled_csv, na_rep="NaN"); print(f"✓ saved {filled_csv.name}")

    # save detail log
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(
            OUT_DIR / f"Station{sta}_verylonggap_fill_detail.csv", index=False)
        print("✓ detail file written")
    else:
        print("• no gaps filled; no detail file")

# ---------- CLI -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int, help="single station id")
    parser.add_argument("--param",   type=str, help="single parameter (SWC_5 etc.)")
    args = parser.parse_args()

    STATIONS = [1,2,3,4,5,6] if args.station is None else [args.station]
    PARAMS   = ["SWC_5","SWC_10","SWC_20","SWC_50"] if args.param is None else [args.param]
    donors_all = [1,2,3,4,5,6]  

    for sid in STATIONS:
        fill_station(sid, PARAMS, donors_all)

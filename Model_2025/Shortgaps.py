# -----------------------------------------------------------
# Shortgaps.py  –  fill every <24‑hour gap via time interpolation
# -----------------------------------------------------------
#   python Shortgaps.py                             # ALL stations & ALL SWC columns
#   python Shortgaps.py --station 2                 # only Station 2
#   python Shortgaps.py --param SWC_10              # all stations, only SWC_10
#   python Shortgaps.py --station 3 --param SWC_50  # just that combo
# -----------------------------------------------------------


# Import libraries
import argparse, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path
CLEAN_DIR = Path("../EDA_2025/cleaned_data")
MISS_DIR  = Path("../EDA_2025/missing_data")
OUT_DIR   = Path("./output")


# 1.Load the data
def load_cleaned_data(station_id, directory="../EDA_2025/cleaned_data"):
    filename = f"{directory}/Station{station_id}_cleaned_data.csv"
    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df.index.freq = 'H'
    return df

def load_missing_data(station_id, directory="../EDA_2025/missing_data"):
    filename = f"{directory}/Station{station_id}_missing_data.csv"
    df = pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])
    return df

# 2.Filter Short gap data (Hours to Days) (<24 hours)
def filter_short_gaps(gap_df, parameter, max_gap = 24):
    gap_df["Number Missing"] = pd.to_numeric(gap_df["Number Missing"], errors="coerce")
    mask = (gap_df["Parameter"] == parameter) & (gap_df["Number Missing"] < max_gap)
    return gap_df[mask].copy()

# 3. Interpolation
def time_interpolate(series, start_ts, end_ts, method="pchip"):
    s = series.copy()
    s[start_ts:end_ts] = np.nan
    s = s.interpolate(method=method)
    return s.loc[start_ts:end_ts]

# 4. Fill short gaps
def fill_short_gaps(series, gap_df):
    filled = series.copy()
    for _, row in gap_df.iterrows():
        start_ts = pd.to_datetime(row["Start Timestamp"])
        end_ts   = pd.to_datetime(row["End Timestamp"])
        filled.loc[start_ts:end_ts] = time_interpolate(filled, start_ts, end_ts)
    return filled

# 5. Process
def process_station(station_id, parameters, save_dir):
    df   = load_cleaned_data(station_id)
    gaps = load_missing_data(station_id)

    any_filled = False
    for param in parameters:
        sg = filter_short_gaps(gaps, param)
        if sg.empty:
            print(f"  {param}: no <24h gaps")
            continue
        print(f"  {param}: filling {len(sg)} gaps")
        df[param] = fill_short_gaps(df[param], sg)
        any_filled = True

    save_dir.mkdir(exist_ok=True)
    out = save_dir / f"Station{station_id}_filled_shortgaps.csv"
    df.to_csv(out)
    if any_filled:
        print(f"→ saved {out}\n")
    else:
        print(f"→ saved {out} (unchanged; nothing to fill)\n")


# Discover all station Ids
def discover_stations():
    pat = re.compile(r"Station(\d+)_cleaned_data\.csv")
    ids = [int(pat.match(fn.name).group(1))
           for fn in CLEAN_DIR.glob("Station*_cleaned_data.csv")
           if pat.match(fn.name)]
    return sorted(ids)

# CLI
def parse_args():
    p = argparse.ArgumentParser("Fill <24 h gaps for one/all stations.")
    p.add_argument("--station", type=int, nargs="*", default=None,
                   help="Station IDs (omit for all discovered).")
    p.add_argument("--param", type=str, nargs="*", default=None,
                   help="SWC columns (omit for SWC_5 10 20 50).")
    return p.parse_args()


# Main function
def main():
    args = parse_args()
    stations   = args.station if args.station else discover_stations()
    parameters = args.param   if args.param else ["SWC_5","SWC_10","SWC_20","SWC_50"]

    print("Stations :", stations)
    print("Parameters:", parameters, "\n")

    for sid in stations:
        print(f"=== Station {sid} ===")
        process_station(sid, parameters, save_dir=OUT_DIR)


if __name__ == "__main__":
    main()
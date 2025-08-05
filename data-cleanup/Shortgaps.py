# -----------------------------------------------------------
# Shortgaps.py  –  fill every <24‑hour gap via time interpolation
# -----------------------------------------------------------
#   python Shortgaps.py                             # ALL stations & ALL SWC columns
#   python Shortgaps.py --station 2                 # only Station 2
#   python Shortgaps.py --param SWC_10              # all stations, only SWC_10
#   python Shortgaps.py --station 3 --param SWC_50  # just that combo
#
# Output files (per station)
#    output/StationX_filled_shortgaps.csv            – cleaned series after filling
#    output/StationX_shortgap_fill_detail.csv        – long‑form table of every value written
# -----------------------------------------------------------


# Import libraries
import argparse, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path
BASE_DIR  = Path(__file__).resolve().parent    
CLEAN_DIR = BASE_DIR / "cleaned_data"
MISS_DIR  = BASE_DIR / "missing_data"
OUT_DIR   = BASE_DIR / "output"


# 1.Load the data
def load_cleaned_data(station_id, directory=CLEAN_DIR):
    filename = Path(directory) / f"Station{station_id}_cleaned_data.csv"
    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    df.index.freq = 'H'
    return df

def load_missing_data(station_id, directory=MISS_DIR):
    filename = Path(directory) / f"Station{station_id}_missing_data.csv"
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
def fill_short_gaps(series, gap_df, gap_log, *, station_id, param):
    filled = series.copy()
    for _, row in gap_df.iterrows():
        start_ts = pd.to_datetime(row["Start Timestamp"])
        end_ts   = pd.to_datetime(row["End Timestamp"])
        idx      = pd.date_range(start_ts, end_ts, freq="H")

        new_vals = time_interpolate(filled, start_ts, end_ts)
        filled.loc[idx] = new_vals.values   # write back

        # --- log each written value ---
        gap_log.extend({
            "Station":   station_id,
            "Parameter": param,
            "Start":     start_ts,
            "End":       end_ts,
            "Timestamp": ts,
            "Filled":    val
        } for ts, val in zip(idx, new_vals.values))
    return filled

# 5. Process
def process_station(station_id, parameters):
    df        = load_cleaned_data(station_id)
    gap_table = load_missing_data(station_id)
    gap_log   = []                              

    any_filled = False
    for param in parameters:
        sgaps = filter_short_gaps(gap_table, param)
        if sgaps.empty:
            print(f"  {param}: no <24h gaps")
            continue
        print(f"  {param}: filling {len(sgaps)} gaps")
        df[param] = fill_short_gaps(df[param], sgaps, gap_log, station_id=station_id, param=param)
        any_filled = True

    OUT_DIR.mkdir(exist_ok=True)
    filled_csv = OUT_DIR / f"Station{station_id}_filled_shortgaps.csv"
    df.to_csv(filled_csv)

    if gap_log:
        detail_csv = OUT_DIR / f"Station{station_id}_shortgap_fill_detail.csv"
        pd.DataFrame(gap_log).to_csv(detail_csv, index=False)
        print(f"    detailed log  →  {detail_csv}")

    status = "(unchanged; nothing to fill)" if not any_filled else ""
    print(f"→ saved {filled_csv} {status}\n")


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
        process_station(sid, parameters)


if __name__ == "__main__":
    main()
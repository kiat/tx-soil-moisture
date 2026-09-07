# -----------------------------------------------------------
# Shortgaps.py  -  fill every <24-hour soil gap with bracketed interpolation
# -----------------------------------------------------------
#   python Shortgaps.py                                     # ALL stations & ALL SWC/T columns
#   python Shortgaps.py --station 2                         # only Station 2 (SWC + T)
#   python Shortgaps.py --param SWC_10 T_20                 # all stations, only these columns
#   python Shortgaps.py --station 3 --param SWC_50 T_10     # just that combo
#
# Output files (per station)
#    output/StationX_filled_shortgaps.csv            – cleaned series after filling
#    output/StationX_shortgap_fill_detail.csv        – long‑form table of every value written
# Notes
#- Soil moisture (SWC_5/10/20/50) uses PCHIP interpolation (existing behavior).
#- Soil temperature (T_5/10/20/50) uses time-based interpolation for smoother diurnal shape.
# -----------------------------------------------------------



# Import libraries
import argparse, re
import pandas as pd
import numpy as np
from pathlib import Path

from param_config import ALL_SOIL_PARAMS, short_interp_for
from time_index_utils import require_unique_datetime_index
from soil_source_coverage import load_soil_coverage

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
    require_unique_datetime_index(df, str(filename))
    df.index.freq = "h"
    return df

def load_missing_data(station_id, directory=MISS_DIR, coverage=None):
    filename = Path(directory) / f"Station{station_id}_missing_data.csv"
    df = pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])
    return (coverage or load_soil_coverage(station_id)).restrict_gaps(df)

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


def wind_direction_interpolate(series, start_ts, end_ts):
    idx = pd.date_range(start_ts, end_ts, freq="h")
    radians = np.deg2rad(series)
    sin_series = np.sin(radians)
    cos_series = np.cos(radians)
    sin_interp = time_interpolate(sin_series, start_ts, end_ts, method="time")
    cos_interp = time_interpolate(cos_series, start_ts, end_ts, method="time")
    angles = (np.degrees(np.arctan2(sin_interp, cos_interp)) + 360) % 360
    return pd.Series(angles, index=idx)

# 4. Fill short gaps
def fill_short_gaps(series, gap_df, gap_log, *, station_id, param, interp_method="pchip", coverage=None):
    coverage = coverage or load_soil_coverage(station_id)
    filled = series.copy()
    for _, row in gap_df.iterrows():
        start_ts = pd.to_datetime(row["Start Timestamp"])
        end_ts   = pd.to_datetime(row["End Timestamp"])
        coverage.require_interval(start_ts, end_ts, param)
        idx      = pd.date_range(start_ts, end_ts, freq="h")

        left_ts = start_ts - pd.Timedelta(hours=1)
        right_ts = end_ts + pd.Timedelta(hours=1)
        bracketed = (
            left_ts in filled.index
            and right_ts in filled.index
            and pd.notna(filled.loc[left_ts])
            and pd.notna(filled.loc[right_ts])
        )
        if not bracketed:
            print(f"    [skip] {param} {start_ts} to {end_ts}: gap is not bracketed by observations")
            continue

        if interp_method == "zero":
            new_vals = pd.Series(0.0, index=idx)
        elif interp_method == "wind_angle":
            new_vals = wind_direction_interpolate(filled, start_ts, end_ts)
        else:
            method = interp_method if interp_method != "time" else "time"
            new_vals = time_interpolate(filled, start_ts, end_ts, method=method)

        if new_vals.isna().any():
            print(f"    [skip] {param} {start_ts} to {end_ts}: interpolation returned NaN")
            continue

        filled.loc[idx] = new_vals.values   # write back

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
    coverage = load_soil_coverage(station_id)
    coverage.assert_frame(df)
    gap_table = load_missing_data(station_id, coverage=coverage)
    gap_log   = []                              

    any_filled = False
    for param in parameters:
        if param not in df.columns:
            print(f"  {param}: column missing in cleaned data – skip")
            continue
        sgaps = filter_short_gaps(gap_table, param)
        if sgaps.empty:
            print(f"  {param}: no <24h gaps")
            continue
        interp_method = short_interp_for(param)

        print(f"  {param}: filling {len(sgaps)} gaps (method={interp_method})")
        before = len(gap_log)
        df[param] = fill_short_gaps(
            df[param], sgaps, gap_log,
            station_id=station_id, param=param, interp_method=interp_method, coverage=coverage
        )
        any_filled = any_filled or len(gap_log) > before

    OUT_DIR.mkdir(exist_ok=True)
    coverage.assert_frame(df)
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
    pat = re.compile(r"Station(.+)_cleaned_data\.csv")
    ids = [pat.match(fn.name).group(1)
           for fn in CLEAN_DIR.glob("Station*_cleaned_data.csv")
           if pat.match(fn.name)]
    return sorted(ids)

# CLI
def parse_args():
    p = argparse.ArgumentParser("Fill <24 h gaps for one/all stations.")
    p.add_argument("--station", type=str, nargs="*", default=None,
                   help="Station IDs/site codes (omit for all discovered).")
    p.add_argument("--param", type=str, nargs="*", default=None,
                   help="Columns to fill. Omit for SWC_5/10/20/50 and T_5/10/20/50.")
    return p.parse_args()


# Main function
def main():
    args = parse_args()
    stations   = args.station if args.station else discover_stations()
    # The main staged workflow is soil-only. MET variables are handled by
    # MetGaps.py so their outputs and validation remain separate.
    default_params = ALL_SOIL_PARAMS
    parameters = args.param   if args.param else default_params
    unsupported = sorted(set(parameters) - set(ALL_SOIL_PARAMS))
    if unsupported:
        raise ValueError(
            "Shortgaps.py is the soil stage. Use MetGaps.py for MET parameters: "
            + ", ".join(unsupported)
        )

    print("Stations :", stations)
    print("Parameters:", parameters, "\n")

    for sid in stations:
        print(f"=== Station {sid} ===")
        process_station(sid, parameters)


if __name__ == "__main__":
    main()

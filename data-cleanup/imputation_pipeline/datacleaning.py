'''python datacleaning.py --station 1'''
'''python datacleaning.py --station 2'''
'''python datacleaning.py --station 3'''
'''python datacleaning.py --station 4'''
'''python datacleaning.py --station 5'''
'''python datacleaning.py --station 6'''

# Import necessary libraries
import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Manual overrides for stations whose sensors reported bad-but-present readings
# that we want to treat as contiguous missing intervals.
MANUAL_GAP_RULES = {
    5: [
        {
            "parameters": ["T_20"],
            "start": "2015-01-01 00:00:00",
            "end": "2016-02-20 12:00:00",
        },
        {
            "parameters": ["T_5", "T_10", "T_20", "T_50"],
            "start": "2018-04-14 20:00:00",
            "end": "2018-05-15 08:00:00",
        },
    ]
}

def load_soil_data(station_id, base_dir):
    """Load soil station data (.dat) and return a datetime‐indexed DataFrame."""
    file_path = Path(base_dir) / f"SM_{station_id}.dat"
    df = pd.read_csv(file_path, sep=",")
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y %H:%M', errors='coerce')
    df = df.set_index('Date')
    df.columns = df.columns.str.strip()
    for col in ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50',
                'T_5', 'T_10', 'T_20', 'T_50', 'Ppt', 'Flag']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_met_data(station_id, base_dir):
    """Load MET station data (.dat) and return a datetime‐indexed DataFrame."""
    file_path = Path(base_dir) / f"MET_{station_id}.dat"
    df = pd.read_csv(
        file_path,
        parse_dates=['Date'],
        index_col='Date',
        date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M', errors='coerce'),
        dtype={
            'Ppt': np.float64,
            'Tair': np.float64,
            'RH': np.float64,
            'Wind speed': np.float64,
            'Wind direction': np.float64,
            'Srad': np.float64
        }
    )
    df.columns = df.columns.str.strip()
    df = df.loc['2015-01-01 00:00:00':]
    for col in ['Ppt', 'Tair', 'RH', 'Wind speed', 'Wind direction', 'Srad']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def merge_raw_data(station_id, soil_base_dir, met_base_dir):
    """Merge soil and MET data, keeping MET precipitation when both exist."""
    df_soil = load_soil_data(station_id, soil_base_dir)
    df_met = load_met_data(station_id, met_base_dir)
    merged = pd.merge(df_soil, df_met,
                      how='inner',
                      left_index=True,
                      right_index=True,
                      suffixes=('_soil', '_met'))
    if 'Ppt_soil' in merged.columns and 'Ppt_met' in merged.columns:
        merged['Ppt'] = merged['Ppt_met']
        merged.drop(columns=['Ppt_soil', 'Ppt_met'], inplace=True)
            
    return merged


def save_merged_data(df, station_id, output_dir):
    """Save merged raw data to CSV; return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"raw_merged_station_{station_id}.csv"
    df.to_csv(out_path)
    return str(out_path)


def find_missing_data(df: pd.DataFrame) -> dict:
    """Return a dict of columns → list of timestamps where data is NaN."""
    missing = {}
    for col in df.columns:
        idx = df.index[df[col].isnull()]
        if not idx.empty:
            missing[col] = idx.tolist()
    return missing


def find_and_replace_wrong_data(df):
    """
    Identify values outside valid ranges, replace them with NaN,
    and return (corrected_df, dict of col → invalid timestamps).
    """
    wrong = {}
    dfc = df.copy()

    # Soil Moisture: between 0 and 0.6
    for col in ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']:
        if col in dfc.columns:
            bad = dfc.index[(dfc[col] < 0) | (dfc[col] > 0.6)]
            if not bad.empty:
                wrong[col] = bad.tolist()
                dfc.loc[bad, col] = np.nan

    # Precipitation: non-negative
    if 'Ppt' in dfc.columns:
        bad = dfc.index[dfc['Ppt'] < 0]
        if not bad.empty:
            wrong['Ppt'] = bad.tolist()
            dfc.loc[bad, 'Ppt'] = np.nan

    # RH: between 0 and 100%
    if 'RH' in dfc.columns:
        bad = dfc.index[(dfc['RH'] < 0) | (dfc['RH'] > 100)]
        if not bad.empty:
            wrong['RH'] = bad.tolist()
            dfc.loc[bad, 'RH'] = np.nan

    # Wind speed: 0 to 25 m/s
    if 'Wind speed' in dfc.columns:
        bad = dfc.index[(dfc['Wind speed'] < 0) | (dfc['Wind speed'] > 25)]
        if not bad.empty:
            wrong['Wind speed'] = bad.tolist()
            dfc.loc[bad, 'Wind speed'] = np.nan

    # Wind direction: 0 to 360°
    if 'Wind direction' in dfc.columns:
        bad = dfc.index[(dfc['Wind direction'] < 0) | (dfc['Wind direction'] > 360)]
        if not bad.empty:
            wrong['Wind direction'] = bad.tolist()
            dfc.loc[bad, 'Wind direction'] = np.nan

    # Solar radiation: must be non-negative
    if 'Srad' in dfc.columns:
        bad = dfc.index[dfc['Srad'] < 0]
        if not bad.empty:
            wrong['Srad'] = bad.tolist()
            dfc.loc[bad, 'Srad'] = np.nan

    # Soil Temperature and Air Temperature: -30°C to 60°C
    for col in ['T_5', 'T_10', 'T_20', 'T_50', 'Tair']:
        if col in dfc.columns:
            bad = dfc.index[(dfc[col] < -30) | (dfc[col] > 60)]
            if not bad.empty:
                wrong[col] = bad.tolist()
                dfc.loc[bad, col] = np.nan

    return dfc, wrong


def combine_nan_lists(missing, wrong):
    """Merge the two dicts of timestamp lists and sort each list."""
    combined = {}
    for key in set(missing) | set(wrong):
        combined[key] = sorted(missing.get(key, []) + wrong.get(key, []))
    return combined


def group_consecutive_dates(dates: list, freq: pd.Timedelta) -> list:
    """Group timestamps that are within 1.5×freq of each other."""
    if not dates:
        return []
    groups, current = [], [dates[0]]
    for prev, curr in zip(dates, dates[1:]):
        # If the time difference is less than or equal to 1.5 times the expected frequency,
        # consider the dates as consecutive.
        if curr - prev <= freq * 1.5:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]
    groups.append(current)
    return groups


def create_missing_summary_df(info):
    """Build a summary DataFrame from the merged NaN/invalid timestamp info."""
    rows = []
    for param, ts_list in info.items():
        for grp in group_consecutive_dates(sorted(ts_list), pd.Timedelta(hours=1)):
            rows.append({
                "Parameter": param,
                "Start Timestamp": grp[0],
                "End Timestamp": grp[-1],
                "Number Missing": len(grp)
            })
    return pd.DataFrame(rows)


def inject_manual_gaps(station_id: int, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Append known bad stretches that should be treated as missing gaps."""
    rules = MANUAL_GAP_RULES.get(station_id)
    if not rules:
        return summary_df

    manual_rows = []
    df = summary_df.copy()
    for rule in rules:
        start = pd.Timestamp(rule["start"])
        end = pd.Timestamp(rule["end"])
        hours = int((end - start) / pd.Timedelta(hours=1)) + 1
        for param in rule["parameters"]:
            overlap = (
                (df["Parameter"] == param)
                & (df["Start Timestamp"] <= end)
                & (df["End Timestamp"] >= start)
            )
            df = df[~overlap]
            manual_rows.append({
                "Parameter": param,
                "Start Timestamp": start,
                "End Timestamp": end,
                "Number Missing": hours,
            })

    if manual_rows:
        df = pd.concat([df, pd.DataFrame(manual_rows)], ignore_index=True)
        df = df.sort_values(["Parameter", "Start Timestamp"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate merged, missing/invalid summary, and cleaned full‐timeline CSVs for a soil station."
    )
    parser.add_argument("--station", "-s", type=int, default=1, help="Station ID (1–6).")
    parser.add_argument("--soil-base-dir", type=str, default="../../datasets/TX-Data/soil_station", help="Path to soil .dat files.")
    parser.add_argument("--met-base-dir", type=str, default="../../datasets/TX-Data/met_station", help="Path to MET .dat files.")
    
    # --- MODIFIED OUTPUT PATHS: Use local directories ---
    parser.add_argument("--raw-output-dir", type=str, default="raw_merged_data", help="Directory for merged CSVs.")
    parser.add_argument("--missing-output", type=str, default=None, help="Filename for missing/invalid summary CSV.")
    parser.add_argument("--cleaned-output", type=str, default=None, help="Filename for cleaned full‐timeline CSV.")
    args = parser.parse_args()

    station_id = args.station

    # Merge and save raw data
    merged_df = merge_raw_data(station_id, args.soil_base_dir, args.met_base_dir)
    raw_path = save_merged_data(merged_df, station_id, args.raw_output_dir)
    print(f"Saved merged data to: {raw_path}")

    # Find missing & invalid, then summarize
    missing = find_missing_data(merged_df)
    corrected_df, wrong = find_and_replace_wrong_data(merged_df)
    combined = combine_nan_lists(missing, wrong)
    summary_df = create_missing_summary_df(combined)
    summary_df = inject_manual_gaps(station_id, summary_df)

    miss_out = args.missing_output or f"missing_data/Station{station_id}_missing_data.csv"
    os.makedirs(Path(miss_out).parent, exist_ok=True)
    summary_df.to_csv(miss_out, index=False)
    print(f"Missing/invalid summary saved to: {miss_out}")

    # Build cleaned full‐timeline and save
    corrected_df.index = pd.to_datetime(corrected_df.index)
    
    full_idx = pd.date_range(start=merged_df.index.min(), end=merged_df.index.max(), freq='H')
    full_df = corrected_df.reindex(full_idx)
    clean_out = args.cleaned_output or f"cleaned_data/Station{station_id}_cleaned_data.csv"
    os.makedirs(Path(clean_out).parent, exist_ok=True)
    full_df.to_csv(clean_out, na_rep="NaN")
    print(f"Cleaned full‐timeline data saved to: {clean_out}")


if __name__ == "__main__":
    main()
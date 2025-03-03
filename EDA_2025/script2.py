'''python script2.py --station 1 --output Station1_missing_data.csv --cleaned Station1_cleaned_data.csv'''
'''python script2.py --station 2 --output Station2_missing_data.csv --cleaned Station2_cleaned_data.csv'''
'''python script2.py --station 3 --output Station3_missing_data.csv --cleaned Station3_cleaned_data.csv'''
'''python script2.py --station 4 --output Station4_missing_data.csv --cleaned Station4_cleaned_data.csv'''
'''python script2.py --station 5 --output Station5_missing_data.csv --cleaned Station5_cleaned_data.csv'''
'''python script2.py --station 6 --output Station6_missing_data.csv --cleaned Station6_cleaned_data.csv'''

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

def load_merged_data(station_id, base_dir="raw_merged_data"):
    file_path = Path(base_dir) / f"raw_merged_station_{station_id}.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def find_missing_data(df):
    missing_info = {}
    for col in df.columns:
        missing_dates = df.index[df[col].isnull()].tolist()
        if missing_dates:
            missing_info[col] = missing_dates
    return missing_info

def find_and_replace_wrong_data(df):
    wrong_info = {}
    df_corrected = df.copy()
    
    # Soil Moisture: between 0 and 0.6
    swc_cols = ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']
    for col in swc_cols:
        if col in df_corrected.columns:
            wrong_idx = df_corrected.index[(df_corrected[col] < 0) | (df_corrected[col] > 0.6)].tolist()
            if wrong_idx:
                wrong_info[col] = wrong_idx
                df_corrected.loc[(df_corrected[col] < 0) | (df_corrected[col] > 0.6), col] = np.nan

    # Precipitation: non-negative
    if 'Ppt' in df_corrected.columns:
        wrong_idx = df_corrected.index[df_corrected['Ppt'] < 0].tolist()
        if wrong_idx:
            wrong_info['Ppt'] = wrong_idx
            df_corrected.loc[df_corrected['Ppt'] < 0, 'Ppt'] = np.nan

    # RH: between 0 and 100%
    if 'RH' in df_corrected.columns:
        wrong_idx = df_corrected.index[(df_corrected['RH'] < 0) | (df_corrected['RH'] > 100)].tolist()
        if wrong_idx:
            wrong_info['RH'] = wrong_idx
            df_corrected.loc[(df_corrected['RH'] < 0) | (df_corrected['RH'] > 100), 'RH'] = np.nan

    # Wind speed: 0 to 25 m/s
    if 'Wind speed' in df_corrected.columns:
        wrong_idx = df_corrected.index[(df_corrected['Wind speed'] < 0) | (df_corrected['Wind speed'] > 25)].tolist()
        if wrong_idx:
            wrong_info['Wind speed'] = wrong_idx
            df_corrected.loc[(df_corrected['Wind speed'] < 0) | (df_corrected['Wind speed'] > 25), 'Wind speed'] = np.nan

    # Wind direction: 0 to 360°
    if 'Wind direction' in df_corrected.columns:
        wrong_idx = df_corrected.index[(df_corrected['Wind direction'] < 0) | (df_corrected['Wind direction'] > 360)].tolist()
        if wrong_idx:
            wrong_info['Wind direction'] = wrong_idx
            df_corrected.loc[(df_corrected['Wind direction'] < 0) | (df_corrected['Wind direction'] > 360), 'Wind direction'] = np.nan

    # Solar radiation: must be non-negative
    if 'Srad' in df_corrected.columns:
        wrong_idx = df_corrected.index[df_corrected['Srad'] < 0].tolist()
        if wrong_idx:
            wrong_info['Srad'] = wrong_idx
            df_corrected.loc[df_corrected['Srad'] < 0, 'Srad'] = np.nan

    # Soil Temperature: -30°C to 60°C
    temp_cols = ['T_5', 'T_10', 'T_20', 'T_50']
    for col in temp_cols:
        if col in df_corrected.columns:
            wrong_idx = df_corrected.index[(df_corrected[col] < -30) | (df_corrected[col] > 60)].tolist()
            if wrong_idx:
                wrong_info[col] = wrong_info.get(col, []) + wrong_idx
                df_corrected.loc[(df_corrected[col] < -30) | (df_corrected[col] > 60), col] = np.nan

    # Air Temperature: -30°C to 60°C
    if 'Tair' in df_corrected.columns:
        wrong_idx = df_corrected.index[(df_corrected['Tair'] < -30) | (df_corrected['Tair'] > 60)].tolist()
        if wrong_idx:
            wrong_info['Tair'] = wrong_idx
            df_corrected.loc[(df_corrected['Tair'] < -30) | (df_corrected['Tair'] > 60), 'Tair'] = np.nan

    
    return df_corrected, wrong_info

def combine_nan_lists(missing_info, wrong_info):
    combined = {}
    all_cols = set(missing_info.keys()).union(wrong_info.keys())
    for col in all_cols:
        combined[col] = sorted(missing_info.get(col, []) + wrong_info.get(col, []))
    return combined

def group_consecutive_dates(dates, freq=pd.Timedelta(hours=1)):
    groups = []
    if not dates:
        return groups
    current_group = [dates[0]]
    for prev, curr in zip(dates, dates[1:]):
        # If the time difference is less than or equal to 1.5 times the expected frequency,
        # consider the dates as consecutive.
        if curr - prev <= freq * 1.5:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    groups.append(current_group)
    return groups

def create_missing_summary_df(missing_info):
    summary_rows = []
    for param, dates in missing_info.items():
        sorted_dates = sorted(dates)
        groups = group_consecutive_dates(sorted_dates, freq=pd.Timedelta(hours=1))
        for group in groups:
            start_ts = group[0]
            end_ts = group[-1]
            count = len(group)
            summary_rows.append({
                "Parameter": param,
                "Start Timestamp": start_ts,
                "End Timestamp": end_ts,
                "Number Missing": count
            })
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV summary of missing data (including wrong values replaced) for a given station."
    )
    parser.add_argument(
        "--station", "-s", type=int, default=1,
        help="Station ID (1-6) for which to generate the missing data summary (default: 1)."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output CSV filename. Defaults to 'Station{station}_missing_data.csv'."
    )
    parser.add_argument(
        "--cleaned", "-c", type=str, default=None,
        help="Output CSV filename for cleaned data (rows with any NaN removed). Defaults to 'Station{station}_cleaned_data.csv'."
    )
    
    args = parser.parse_args()
    station_id = args.station
    missing_data_dir = Path("missing_data")
    missing_data_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_filename = missing_data_dir / args.output
    else:
        output_filename = missing_data_dir / f"Station{station_id}_missing_data.csv"
    
    # By default, place the cleaned data CSV in "cleaned_data" folder
    cleaned_data_dir = Path("cleaned_data")
    cleaned_data_dir.mkdir(parents=True, exist_ok=True)
    if args.cleaned:
        cleaned_filename = cleaned_data_dir / args.cleaned
    else:
        cleaned_filename = cleaned_data_dir / f"Station{station_id}_full_timeline.csv"

    # Load merged data for the specified station
    df_merged = load_merged_data(station_id)
    
    # Get missing data info from merged data
    missing_info = find_missing_data(df_merged)

    # Get wrong data info (using a copy so as not to modify the original)
    df_corrected, wrong_info = find_and_replace_wrong_data(df_merged.copy())
    
    # Always combine missing and wrong data info
    combined_info = combine_nan_lists(missing_info, wrong_info)
    
    # Create the summary DataFrame
    summary_df = create_missing_summary_df(combined_info)
    
    # Save the summary DataFrame to CSV
    summary_df.to_csv(output_filename, index=False)
    print(f"Missing data summary saved to: {output_filename}")

    all_dates = pd.date_range(
        start=df_merged.index.min(),
        end=df_merged.index.max(),
        freq='H'
    )
    df_full = df_corrected.reindex(all_dates)
    df_full.to_csv(cleaned_filename, na_rep="NaN")
    print(f"Full timeline Cleaned data saved to: {cleaned_filename}")

if __name__ == "__main__":
    main()
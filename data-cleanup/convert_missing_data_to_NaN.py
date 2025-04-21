import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

#combination of script 1 and 2

# creates a raw merged dataset

# Get base directories from environment variables or use defaults
project_root = Path(__file__).resolve().parent.parent  # goes to tx-soil-moisture/
sm_base_dir = Path(os.getenv("SOIL_DATA_DIR", project_root / "datasets/TX-Data/soil_station"))
met_base_dir = Path(os.getenv("MET_DATA_DIR", project_root / "datasets/TX-Data/met_station"))

# Global dictionary to store cleaned data in memory
cleaned_data_cache = {}

def load_soil_data(file_name, base_dir=sm_base_dir):
    file_path = Path(base_dir) / f"{file_name}.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep=",", na_values=["", "NA", "null"])

    # Convert Date to datetime format and set as index
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y %H:%M', errors='coerce')

    df = df.set_index('Date')
    
    # Remove spaces
    df.columns = df.columns.str.strip()
    
    # Drop the 'Flag' column if it exists
    if 'Flag' in df.columns:
        df = df.drop(columns=['Flag'])
    
    # Convert datatype to float
    numeric_cols = ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50', 
                    'T_5', 'T_10', 'T_20', 'T_50', 'Ppt']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def load_met_data(file_name, base_dir=met_base_dir):
    file_path = Path(base_dir) / f"{file_name}.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep=",", na_values=["", "NA", "null"])

    # Convert Date to datetime format and set as index
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y %H:%M', errors='coerce')

    df = df.set_index('Date')
    
    # Remove spaces
    df.columns = df.columns.str.strip()
    
    # Convert datatype to float
    numeric_cols = ['T_air', 'RH', 'Ppt', 'Flag']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def save_cleaned_data(df, station_id, output_dir="raw_merged_data"):
    # Save the DataFrame to the cache
    cleaned_data_cache[station_id] = df

    # Save the DataFrame to a file
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/raw_merged_station_{station_id}.csv"
    df.to_csv(output_path, na_rep='NaN')
    print(f"Saved cleaned data to: {output_path}")


############# script 2 below
def load_merged_data(station_id, df=None):
    if df is not None:
        return df
    elif station_id in cleaned_data_cache:
        # Use the cached DataFrame if available
        return cleaned_data_cache[station_id]
    else:
        # Load from file if no DataFrame is provided and not in cache
        base_dir = "raw_merged_data"
        file_path = Path(base_dir) / f"raw_merged_station_{station_id}.csv"
        df = pd.read_csv(file_path, index_col=0, parse_dates=True, na_values=["", "NA", "null"])
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
        # Ensure the column is numeric
        df_corrected['Wind speed'] = pd.to_numeric(df_corrected['Wind speed'], errors='coerce')
        
        # Find invalid values
        wrong_idx = df_corrected.index[(df_corrected['Wind speed'] < 0) | (df_corrected['Wind speed'] > 25)].tolist()
        if wrong_idx:
            wrong_info['Wind speed'] = wrong_idx
            df_corrected.loc[(df_corrected['Wind speed'] < 0) | (df_corrected['Wind speed'] > 25), 'Wind speed'] = np.nan

    # Wind direction: 0 to 360°
    if 'Wind direction' in df_corrected.columns:
        # Ensure the column is numeric
        df_corrected['Wind direction'] = pd.to_numeric(df_corrected['Wind direction'], errors='coerce')
        
        # Find invalid values
        wrong_idx = df_corrected.index[(df_corrected['Wind direction'] < 0) | (df_corrected['Wind direction'] > 360)].tolist()
        if wrong_idx:
            wrong_info['Wind direction'] = wrong_idx
            df_corrected.loc[(df_corrected['Wind direction'] < 0) | (df_corrected['Wind direction'] > 360), 'Wind direction'] = np.nan

    # Solar radiation: must be non-negative
    if 'Srad' in df_corrected.columns:
        # Ensure the column is numeric
        df_corrected['Srad'] = pd.to_numeric(df_corrected['Srad'], errors='coerce')
        
        # Find invalid values
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
        # Ensure the column is numeric
        df_corrected['Tair'] = pd.to_numeric(df_corrected['Tair'], errors='coerce')
        
        # Find invalid values
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

def indv_timestamps(df):
    df_missing = df.copy()
    missing_values_list = []
    for col in df_missing.columns:
        missing_rows = df_missing[df_missing[col].isna()]
        for timestamp in missing_rows.index:
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")  # Format timestamp
            missing_values_list.append([formatted_timestamp, col])

    missing_values_df = pd.DataFrame(missing_values_list, columns=["Timestamp", "Parameter"])
    return missing_values_df

def delete_generated_files(station_id=None, output_dirs=["raw_merged_data", "missing_cleaned_data"]):
    """
    Deletes all files in the specified output directories or specific files for a given station.
    """
    for output_dir in output_dirs:
        output_path = Path(output_dir)
        if output_path.exists() and output_path.is_dir():
            if station_id:
                # Delete files specific to the given station
                deleted_files = []
                for file in output_path.iterdir():
                    if file.is_file() and f"station_{station_id}" in file.name:
                        file.unlink()  # Delete the file
                        deleted_files.append(file.name)
                if not deleted_files:
                    print(f"No files found for Station {station_id} in {output_dir}.")
                else:
                    print(f"Deleted files for Station {station_id} in {output_dir}: {deleted_files}")
            else:
                # Delete all files in the directory
                deleted_files = []
                for file in output_path.iterdir():
                    if file.is_file():
                        file.unlink()  # Delete the file
                        deleted_files.append(file.name)
                if not deleted_files:
                    print(f"No files found in {output_dir}.")
                else:
                    print(f"Deleted all files in {output_dir}: {deleted_files}")
        else:
            print(f"Directory {output_dir} does not exist or is empty.")

def process_station(station_id, output_ranges_filename=None, cleaned_filename=None, missing_timestamps_filename=None):
    print(f"Processing Station {station_id}...")

    # Default output directory and filenames
    missing_output_dir = "missing_cleaned_data"
    raw_output_dir = "raw_merged_data"
    os.makedirs(missing_output_dir, exist_ok=True)
    os.makedirs(raw_output_dir, exist_ok=True)

    output_ranges_filename = output_ranges_filename or f"{missing_output_dir}/missing_ranges_station_{station_id}.csv"
    cleaned_filename = cleaned_filename or f"{missing_output_dir}/cleaned_data_station_{station_id}.csv"
    missing_timestamps_filename = missing_timestamps_filename or f"{missing_output_dir}/missing_timestamps_station_{station_id}.csv"
    raw_merged_filename = f"{raw_output_dir}/raw_merged_station_{station_id}.csv"

    # Load and merge raw data
    soil_file = f"SM_{station_id}"
    met_file = f"MET_{station_id}"
    soil_data = load_soil_data(soil_file)
    met_data = load_met_data(met_file)
    df_merged = pd.merge(soil_data, met_data, left_index=True, right_index=True, suffixes=('_soil', '_met'))

    # Save raw merged data
    cleaned_data_cache[station_id] = df_merged
    # df_merged.to_csv(raw_merged_filename, na_rep='NaN')
    # print(f"Raw merged data saved to: {raw_merged_filename}")

    # Detect missing and wrong data
    missing_info = find_missing_data(df_merged)
    df_corrected, wrong_info = find_and_replace_wrong_data(df_merged.copy())

    # Combine missing and wrong data info
    combined_info = combine_nan_lists(missing_info, wrong_info)
    summary_df = create_missing_summary_df(combined_info)
    missing_indv_values = indv_timestamps(df_merged)

    # Save output of the ranges
    # summary_df.to_csv(output_ranges_filename, index=False, na_rep='NaN')
    # print(f"Missing data ranges saved to: {output_ranges_filename}")

    # missing_indv_values.to_csv(missing_timestamps_filename, index=False, na_rep='NaN')
    # print(f"Individual missing timestamps saved to: {missing_timestamps_filename}")

    # Save cleaned dataset with hourly reindexing
    all_dates = pd.date_range(start=df_merged.index.min(), end=df_merged.index.max(), freq='H')
    df_full = df_corrected.reindex(all_dates)  # Retain NaN values without filling them
    df_full.to_csv(cleaned_filename, na_rep='NaN')  # Ensure NaN is explicitly represented in the output
    print(f"Cleaned data saved to: {cleaned_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Process soil and met data, and optionally clean up generated files."
    )
    parser.add_argument(
        "--station", "-s", type=str, default=None,
        help="Station ID (1-6) or 'all' to process all stations."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Shortcut to process or clean all stations."
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete all files created by the script, or specific files for a station."
    )
    args = parser.parse_args()

    # Handle the --all flag
    if args.all:
        args.station = "all"

    if args.clean:
        if args.station == "all":
            delete_generated_files()  # Delete all files
        elif args.station:
            delete_generated_files(station_id=args.station)  # Delete specific station files
        else:
            print("Please specify a station or use --all with --clean.")
        return

    # Normal processing logic here
    station_ids = (
        range(1, 7) if args.station == "all" else [int(args.station)]
    )
    for station_id in station_ids:
        try:
            process_station(station_id)
        except Exception as e:
            print(f"Error processing Station {station_id}: {e}")

if __name__ == "__main__":
    main()
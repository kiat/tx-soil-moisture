import os
import pandas as pd
from pathlib import Path

def load_station_data(config):
    """
    Loads all CSV files from the processed data folder defined in the config.
    Returns a dictionary of DataFrames with station names as keys.
    """
    data_folder = Path(config["paths"]["data_folder"])

    if not data_folder.exists():
        raise FileNotFoundError(f"ERROR: Folder '{data_folder}' does not exist!")

    station_files = list(data_folder.glob("*.csv"))
    
    if not station_files:
        raise FileNotFoundError(f"ERROR: No CSV files found in '{data_folder}'.")

    station_data = {}
    for file_path in station_files:
        station_name = file_path.stem.replace("_Revised_Final_Data", "")

        # Read CSV and ensure timestamp is correctly set as index
        df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
        
        station_data[station_name] = df

    print(f"Loaded data from {len(station_data)} stations.")
    
    return station_data

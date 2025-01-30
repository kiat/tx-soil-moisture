import os
import pandas as pd
from helpers.constants import DATA_FOLDER  # Import constant

def load_station_data():
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"ERROR: Folder '{DATA_FOLDER}' does not exist!")

    station_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

    if not station_files:
        raise FileNotFoundError(f"ERROR: No CSV files found in '{DATA_FOLDER}'.")

    station_data = {}
    for file in station_files:
        station_name = file.replace("_Revised_Final_Data.csv", "")
        file_path = os.path.join(DATA_FOLDER, file)
        df = pd.read_csv(file_path, parse_dates=["Unnamed: 0"], index_col="Unnamed: 0")
        station_data[station_name] = df

    print(f"Loaded data from {len(station_data)} stations.")
    return station_data

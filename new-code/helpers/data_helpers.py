import os
import pandas as pd
from constants import DATA_FOLDER

def load_station_data():
    """
    Loads all CSV files from the processed data folder.
    Returns a dictionary of DataFrames with station names as keys.
    """
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"ERROR: Folder '{DATA_FOLDER}' does not exist!")

    station_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

    if not station_files:
        raise FileNotFoundError(f"ERROR: No CSV files found in '{DATA_FOLDER}'.")

    station_data = {}
    for file in station_files:
        station_name = file.replace("_Revised_Final_Data.csv", "")
        file_path = os.path.join(DATA_FOLDER, file)

        # Read CSV and ensure the first column (timestamp) is correctly set as an index
        df = pd.read_csv(file_path, header=0)
        
        # If the first column has no header, rename it
        if df.columns[0] == "":
            df.rename(columns={df.columns[0]: "Timestamp"}, inplace=True)

        # Convert the first column to datetime and set it as an index
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df.set_index(df.columns[0], inplace=True)

        station_data[station_name] = df

    print(f"Loaded data from {len(station_data)} stations.")
    
    return station_data

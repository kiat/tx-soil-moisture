# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import yaml
from pathlib import Path
import pandas as pd

def load_config():
    config_path = Path(__file__).parent.parent / "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



def load_station_data(config):
    data_folder = Path(config["paths"]["data_folder"])
    if not data_folder.exists():
        raise FileNotFoundError(f"ERROR: Folder '{data_folder}' does not exist!")
    
    station_files = list(data_folder.glob("*.csv"))
    if not station_files:
        raise FileNotFoundError(f"ERROR: No CSV files found in '{data_folder}'.")
    
    station_data = {}
    for file_path in station_files:
        station_name = file_path.stem.replace("_Revised_Final_Data", "")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.columns = df.columns.str.replace(" ", "")
        station_data[station_name] = df
    print(f"Loaded data from {len(station_data)} stations: {list(station_data.keys())}")
    return station_data



def engineer_data(df):
    import numpy as np
    # Consolidate wind vectors only if both 'Windspeed' and 'Winddirection' exist
    if 'Windspeed' in df.columns and 'Winddirection' in df.columns:
        wv = df.pop('Windspeed')
        wd_rad = df.pop('Winddirection') * np.pi / 180
        df['Wx'] = wv * np.cos(wd_rad)
        df['Wy'] = wv * np.sin(wd_rad)
    
    # Remove Latitude and Longitude if they exist
    if 'Latitude' in df.columns:
        df.pop('Latitude')
    if 'Longitude' in df.columns:
        df.pop('Longitude')
    
    # Add periodic time features based on the index (timestamp in seconds)
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    return df


def load_and_engineer_data(config):
    station_data = load_station_data(config)
    for station in station_data:
        station_data[station] = engineer_data(station_data[station])
    return station_data

import numpy as np

def data_to_X_y(series, window_size, offset):
    data = series.values
    X, y = [], []
    for i in range(len(data) - window_size - offset):
        X.append(data[i:i+window_size].reshape(-1, 1))
        y.append(data[i+window_size+offset])
    return np.array(X), np.array(y)

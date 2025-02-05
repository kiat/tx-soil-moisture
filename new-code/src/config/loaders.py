import yaml
from pathlib import Path
import pandas as pd

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
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
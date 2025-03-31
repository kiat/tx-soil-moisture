import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")


# Get base directories from environment variables or use defaults
sm_base_dir = Path(os.getenv("SOIL_DATA_DIR", Path(__file__).resolve().parent.parent.parent / "datasets/TX-Data/soil_station"))
met_base_dir = Path(os.getenv("MET_DATA_DIR", Path(__file__).resolve().parent.parent.parent / "datasets/TX-Data/met_station"))

def load_soil_data(station_id=1, base_dir=sm_base_dir):
    """
    Load soil station data from .dat files.
    
    Args:
        station_id (int): Station ID (1-6).
        base_dir (str): Directory containing soil station files.
    
    Returns:
        pd.DataFrame: Raw soil data with datetime index.
    """
    file_path = Path(base_dir) / f"SM_{station_id}.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep=",")

    # Convert Date to datetime format and set as index
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y %H:%M', errors='coerce')

    df = df.set_index('Date')
    
    # Remove spaces
    df.columns = df.columns.str.strip()
    
    # Convert datatype to float
    numeric_cols = ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50', 
                    'T_5', 'T_10', 'T_20', 'T_50', 'Ppt', 'Flag']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
def load_met_data(station_id=1, base_dir=met_base_dir):
    """Load and preprocess MET station data"""
    file_path = Path(base_dir) / f"MET_{station_id}.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep=",")
    
    # Convert Date to datetime format and set as index
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
    # Remove spaces
    df.columns = df.columns.str.strip()
    
    # Filter data from 2015-01-01
    df = df.loc['2015-01-01 00:00:00':]

    # Convert numeric columns
    numeric_cols = ['Ppt', 'Tair', 'RH', 'Wind speed', 'Wind direction', 'Srad']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
def merge_raw_data(station_id):
    # Load soil and MET raw data
    df_soil = load_soil_data(station_id)
    df_met = load_met_data(station_id)
    
    # Merge on the datetime index
    merged_df = pd.merge(df_soil, df_met, how='inner', left_index=True, right_index=True, 
                         suffixes=('_soil', '_met'))
    
    # For the Ppt column, if both exist, choose one (here we choose the MET version)
    if 'Ppt_soil' in merged_df.columns and 'Ppt_met' in merged_df.columns:
        merged_df['Ppt'] = merged_df['Ppt_met']
        merged_df.drop(columns=['Ppt_soil', 'Ppt_met'], inplace=True)
        
    return merged_df
def save_merged_data(df, station_id, output_dir="raw_merged_data"):
    """
    Save merged data to CSV.
    
    Args:
        df (pd.DataFrame): Merged soil and met data.
        station_id (int): Station ID (1-6).
        output_dir (str): Output directory path.
    """
    # Drop the 'Flag' column if it exists
    if 'Flag' in df.columns:
        df.drop(columns=['Flag'], inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/raw_merged_station_{station_id}.csv"
    df.to_csv(output_path)
    print(f"Saved merged data to: {output_path}")
    return df

for station_id in range(1, 7):
    save_merged_data(merge_raw_data(station_id), station_id)

def find_missing_data(df):
    missing_info = {}
    for col in df.columns:
        missing_dates = df.index[df[col].isnull()].tolist()
        if missing_dates:
            missing_info[col] = missing_dates
    return missing_info

df_merged = pd.read_csv("raw_merged_data/raw_merged_station_1.csv", index_col=0, parse_dates=True)
missing_info = find_missing_data(df_merged)

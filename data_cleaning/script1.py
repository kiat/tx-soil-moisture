import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse
import warnings
warnings.filterwarnings("ignore")


# Get base directories from environment variables or use defaults
sm_base_dir = Path(os.getenv("SOIL_DATA_DIR", Path(__file__).resolve().parent.parent.parent / "datasets/TX-Data/soil_station"))
met_base_dir = Path(os.getenv("MET_DATA_DIR", Path(__file__).resolve().parent.parent.parent / "datasets/TX-Data/met_station"))

def load_soil_data(file_name, base_dir=sm_base_dir):
    """
    Load soil station data from .dat files.
    
    Args:
        file_name (str): File name for the soil station data (e.g., SM_1.dat).
        base_dir (str): Directory containing soil station files.
    
    Returns:
        pd.DataFrame: Raw soil data with datetime index.
    """
    file_path = Path(base_dir) / f"{file_name}.dat"
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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def load_met_data(file_name, base_dir=met_base_dir):
    """
    Load MET station data from .dat files.
    
    Args:
        file_name (str): File name for the MET station data (e.g., MET_1.dat).
        base_dir (str): Directory containing MET station files.
    
    Returns:
        pd.DataFrame: Raw MET data with datetime index.
    """
    file_path = Path(base_dir) / f"{file_name}.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep=",")

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

def save_cleaned_data(df, station_id, output_dir="cleaned_data"):
    """
    Save cleaned data to CSV.
    
    Args:
        df (pd.DataFrame): Cleaned data.
        station_id (int): Station ID.
        output_dir (str): Output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/cleaned_station_{station_id}.csv"
    df.to_csv(output_path)
    print(f"Saved cleaned data to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process soil and met station data.")
    parser.add_argument("soil_file", type=str, help="Soil station file name (e.g., SM_1)")
    parser.add_argument("met_file", type=str, help="MET station file name (e.g., MET_1)")
    parser.add_argument("--station_id", type=int, required=True, help="Station ID (e.g., 1, 2, 3, etc.)")
    args = parser.parse_args()

    # Load data
    soil_data = load_soil_data(args.soil_file)
    met_data = load_met_data(args.met_file)

    # Merge data
    merged_data = pd.merge(soil_data, met_data, left_index=True, right_index=True, suffixes=('_soil', '_met'))

    # Save cleaned data for the specified station
    save_cleaned_data(merged_data, args.station_id)

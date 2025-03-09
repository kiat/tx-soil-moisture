import os
import pandas as pd
import numpy as np

def read_and_save_parquet():
    """Reads CSVs, processes them, and saves as Parquet"""
    dfs = {}
    
    for index in range(6):
        csv_path = f'Station{index + 1}_Revised_Final_Data.csv'
        parquet_path = f'Station{index + 1}_Revised_Final_Data.parquet'
        
        if os.path.exists(parquet_path):
            print(f"Skipping {csv_path}, Parquet already exists: {parquet_path}")
            continue
        
        print(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path, sep=",")
        df.columns = df.columns.str.replace(' ', '')
        df.insert(0, 'Date', pd.to_datetime(df['Unnamed:0'], errors='coerce'))
        df.drop('Unnamed:0', axis=1, inplace=True)
        df.set_index('Date', inplace=True)

        for col in ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50', 'T_5', 'T_10', 'T_20', 'T_50', 'Ppt']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df.to_parquet(parquet_path, engine="fastparquet")
        print(f"Saved to Parquet: {parquet_path}")

    return dfs

def engineer_and_save_data():
    """Loads Parquet files, applies feature engineering, and saves engineered data"""
    for index in range(6):
        station_name = f'Station{index + 1}'
        raw_parquet_path = f"{station_name}_Revised_Final_Data.parquet"
        engineered_parquet_path = f"{station_name}_engineered.parquet"

        if os.path.exists(engineered_parquet_path):
            print(f"Skipping {station_name}, Engineered Parquet already exists: {engineered_parquet_path}")
            continue

        print(f"Processing data for {station_name}")
        df = pd.read_parquet(raw_parquet_path)

        # Check for missing columns
        if 'Windspeed' in df and 'Winddirection' in df:
            wv = df.pop('Windspeed')
            wd_rad = df.pop('Winddirection') * np.pi / 180
            max_wv = np.max(wv)
            df['Wx'] = wv * np.cos(wd_rad)
            df['Wy'] = wv * np.sin(wd_rad)
            df['max Wx'] = max_wv * np.cos(wd_rad)
            df['max Wy'] = max_wv * np.sin(wd_rad)
        else:
            print(f"Skipping wind features for {station_name} (columns missing)")

        # Convert index to timestamp
        timestamp_s = df.index.map(pd.Timestamp.timestamp)
        day, year, month = 24 * 60 * 60, 24 * 3600 * 365.2425, 24 * 60 * 60 * 30.4167
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
        df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))

        df = df.reset_index(drop=True)
        df.to_parquet(engineered_parquet_path, engine="fastparquet")
        print(f"Saved engineered data to Parquet: {engineered_parquet_path}")

# if __name__ == "__main__":
    # read_and_save_parquet()  # Load raw data & save as Parquet
    # engineer_and_save_data()  # Engineer & save processed data
import pandas as pd
import numpy as np

def read_and_process_csvs():
    """Reads and processes CSVs, returns dict of cleaned DataFrames"""
    dfs = {}
    for index in range(6):
        station_name = f'Station{index + 1}'
        csv_path = f'Revised_Final_Data/{station_name}_Revised_Final_Data.csv'
        
        print(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.replace(' ', '')  # Clean column names
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Parse Date column
        df.set_index('Date', inplace=True)

        for col in ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50', 'T_5', 'T_10', 'T_20', 'T_50', 'Ppt']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        dfs[station_name] = df
    return dfs

def engineer_features(dfs):
    """Applies feature engineering and returns new dict of engineered DataFrames"""
    engineered_dfs = {}
    for station_name, df in dfs.items():
        print(f"Engineering features for {station_name}")

        # Wind features
        if 'Windspeed' in df and 'Winddirection' in df:
            wv = df.pop('Windspeed')
            wd_rad = df.pop('Winddirection') * np.pi / 180
            max_wv = np.max(wv)
            df['Wx'] = wv * np.cos(wd_rad)
            df['Wy'] = wv * np.sin(wd_rad)
            df['max_Wx'] = max_wv * np.cos(wd_rad)
            df['max_Wy'] = max_wv * np.sin(wd_rad)
        else:
            print(f"Skipping wind features for {station_name} (columns missing)")

        # Time features
        timestamp_s = df.index.map(pd.Timestamp.timestamp)
        day, year, month = 86400, 86400 * 365.2425, 86400 * 30.4167
        df['DaySin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['DayCos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['YearSin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['YearCos'] = np.cos(timestamp_s * (2 * np.pi / year))
        df['MonthSin'] = np.sin(timestamp_s * (2 * np.pi / month))
        df['MonthCos'] = np.cos(timestamp_s * (2 * np.pi / month))

        df = df.reset_index(drop=False)  # Keep 'Date' column if needed
        engineered_dfs[station_name] = df

    return engineered_dfs

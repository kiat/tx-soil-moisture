import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Read and process CSV files
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


# Feature engineering
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

        df['Date'] = df.index # Keep 'Date' column if needed
        engineered_dfs[station_name] = df

    return engineered_dfs


##########################################
def normalize_features(df, features):
    scaler = MinMaxScaler()

    no_scale_features = [feat for feat in features if 'sin' in feat or 'cos' in feat]
    scale_features = [feat for feat in features if feat not in no_scale_features]

    df = df.reset_index(drop=True)

    scaled_data = scaler.fit_transform(df[scale_features])
    scaled_df = pd.DataFrame(scaled_data, columns=scale_features)

    scaled_df = pd.concat([scaled_df, df[no_scale_features]], axis=1)

    scaled_df = scaled_df[features]

    return scaled_df.to_numpy(), scaler

###########################################

# Convert data to X and y windows with offset

# def data_to_X_y(data, window_size, offset):
#     X, y = [], []
#     for i in range(len(data) - window_size - offset):
#         X.append(data[i:i+window_size, :])  
#         y.append(data[i + window_size + offset, 0])  

#     return  np.array(X),  np.array(y)


def data_to_X_y(data, window_size, offset):
    rows = len(data) - window_size - offset
    X = np.lib.stride_tricks.sliding_window_view(data, (window_size, data.shape[1]))[:rows, 0]
    y = data[window_size + offset : window_size + offset + rows, 0]
    return X, y



###############################################################################

# Split data into train, validation, and test sets
    
def split_and_stack_data(dfs, test_station_name="Station6", remove_met=False):
    if remove_met:
        for key in dfs.keys():
            dfs[key] = dfs[key][["SWC_5", "SWC_10", "SWC_20", "SWC_50"]]

    # Get a clean copy of Station6 data
    test_station_data = dfs[test_station_name].copy()

    # Create test and val splits from Station6
    val_df = test_station_data.loc[:'2019-12-31 23:59:59']
    test_df = test_station_data.loc['2020-01-01 00:00:00':]
    print("VAL DF:", val_df["Date"].min(), "to", val_df["Date"].max())
    print("TEST DF:", test_df["Date"].min(), "to", test_df["Date"].max())
    # Remove Station6 entirely from training
    dfs = {k: v for k, v in dfs.items() if k != test_station_name}

    return dfs, val_df, test_df



#######################################################################################

# Methods for visualizing the data splits

def plot_split_timeline(train_df, val_df, test_df, feature, save_dir="results/data_splits/"):
    os.makedirs(save_dir, exist_ok=True)  # Ensure folder exists

    save_path = os.path.join(save_dir, f"split_{feature}.png")  # Dynamic file name

    plt.figure(figsize=(15, 5))
    plt.plot(train_df.index, train_df[feature], label='Train', color='blue', alpha=0.6)
    plt.plot(val_df.index, val_df[feature], label='Val', color='orange', alpha=0.8)
    plt.plot(test_df.index, test_df[feature], label='Test', color='green', alpha=0.8)
    plt.title(f"{feature} Over Time — Train/Val/Test Split")
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()
    return

def concatenate_with_gaps(dfs):
    gap = pd.DataFrame({col: [np.nan] for col in dfs[0].columns}, index=[pd.Timestamp("2099-01-01")])
    spaced = []
    for df in dfs:
        spaced.append(df)
        spaced.append(gap)
    return pd.concat(spaced)

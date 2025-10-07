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

        #precipitation
        # --- precipitation fixes ---
        df['Ppt'] = df['Ppt'].fillna(0.0)
        df['Ppt_RainFlag'] = (df['Ppt'] > 0).astype(int)
        df['Ppt_log'] = np.log1p(df['Ppt'])
        df['Ppt_3h_sum'] = df['Ppt'].rolling(3, min_periods=1).sum()
        df['Ppt_24h_sum'] = df['Ppt'].rolling(24, min_periods=1).sum()
        # time‑since‑rain:
        hours_since = []
        count = 0
        for v in df['Ppt']:
            count = 0 if v > 0 else count + 1
            hours_since.append(count)
        df['HoursSinceRain'] = hours_since

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


# Normalize features
def normalize_features(df, features):
    scaler = MinMaxScaler()
    # Identify features to scale and not to scale
    no_scale_features = [feat for feat in features if 'sin' in feat or 'cos' in feat]
    scale_features = [feat for feat in features if feat not in no_scale_features]

    # Reset index to avoid issues with scaling
    df = df.reset_index(drop=True)

    # Scale the features
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


# Convert data to X and y windows with offset
def data_to_X_y(data, window_size, offset, label_type="point", agg_hours=24, offset_hours=0, samples_per_hour=1):
    """
    Generate windowed data with flexible label generation.
    
    Args:
        data: Input data array (time_steps, features)
        window_size: Size of input window
        offset: Original offset parameter (kept for backward compatibility)
        label_type: Type of label generation ('point', 'rolling_mean', 'daily_mean')
        agg_hours: Hours to aggregate for rolling_mean (ignored for point and daily_mean)
        offset_hours: Forecast offset in hours (predict average ending at t + offset_hours)
        samples_per_hour: Number of samples per hour in the data
        
    Returns:
        X: Input windows (n_samples, window_size, n_features)
        y: Target labels (n_samples,)
    """
    
    # Convert hours to sample indices
    agg_samples = agg_hours * samples_per_hour
    offset_samples = offset_hours * samples_per_hour
    
    # For backward compatibility, if label_type is "point", use original logic
    if label_type == "point":
        rows = len(data) - window_size - offset
        X = np.lib.stride_tricks.sliding_window_view(data, (window_size, data.shape[1]))[:rows, 0]
        y = data[window_size + offset - 1: window_size + offset - 1 + rows, 0]
        return X, y
    
    # Calculate the total offset from window end to target end
    total_offset = offset + offset_samples
    
    # For rolling_mean and daily_mean, we need to ensure we have enough history
    if label_type == "rolling_mean":
        # Need agg_samples of history for the target
        # target_end_idx will be at window_size + total_offset, so we need data up to that point
        min_required_length = window_size + total_offset + agg_samples
    elif label_type == "daily_mean":
        # For daily_mean, we need at least 24 hours of data (or samples_per_hour * 24)
        min_required_length = window_size + total_offset + (24 * samples_per_hour)
    else:
        raise ValueError(f"Unknown label_type: {label_type}")
    
    if len(data) < min_required_length:
        print(f"Warning: Data length {len(data)} is less than required {min_required_length} for label_type={label_type}")
        return np.array([]), np.array([])
    
    # Calculate number of valid windows
    rows = len(data) - min_required_length + 1
    if rows <= 0:
        return np.array([]), np.array([])
    
    # Generate input windows
    X = np.lib.stride_tricks.sliding_window_view(data, (window_size, data.shape[1]))[:rows, 0]
    
    # Generate labels based on type
    y = []
    
    for i in range(rows):
        # Target end point: end of window + original offset + offset_hours
        # Note: window ends at i + window_size - 1, so target starts at i + window_size
        target_end_idx = i + window_size + total_offset
        
        if label_type == "rolling_mean":
            # Take the mean of the last agg_samples before target_end_idx
            target_start_idx = target_end_idx - agg_samples + 1
            if target_start_idx < 0:
                # Skip this window if insufficient history
                continue
            target_values = data[target_start_idx:target_end_idx + 1, 0]
            y.append(np.mean(target_values))
            
        elif label_type == "daily_mean":
            # For daily_mean, we need to compute calendar day average
            # This is a simplified version - in practice, you might want to use actual timestamps
            # For now, we'll use a 24-hour rolling window (24 * samples_per_hour samples)
            daily_samples = 24 * samples_per_hour
            target_start_idx = target_end_idx - daily_samples + 1
            if target_start_idx < 0:
                continue
            target_values = data[target_start_idx:target_end_idx + 1, 0]
            y.append(np.mean(target_values))
    
    # Trim X to match y length (in case some windows were skipped)
    X = X[:len(y)]
    y = np.array(y)
    
    return X, y




# Split data into train, validation, and test sets
# By default, the test set is the 2020 data in Station6, and the validation set is the rest of Station6
# The training set is all other stations
def split_and_stack_data(dfs, test_station_name="Station6", remove_met=False):
    if remove_met:
        for key in dfs.keys():
            dfs[key] = dfs[key][["SWC_5", "SWC_10", "SWC_20", "SWC_50"]]

    # Get a clean copy of Station6 data
    test_station_data = dfs[test_station_name].copy()
    
    train_val_stations = [k for k in dfs.keys() if k != test_station_name]

    # Split train/val for stations 1-5
    train_dfs, val_dfs = [], []
    for k in train_val_stations:
        df = dfs[k]
        train_dfs.append(df.loc[:'2018-12-31 23:59:59'])
        val_dfs.append(df.loc['2019-01-01 00:00:00':])

    # train_df = pd.concat(train_dfs)
    # val_df = pd.concat(val_dfs)
    # val_df = test_station_data.loc[:'2019-12-31 23:59:59']
    test_df = test_station_data.loc['2020-01-01 00:00:00':]
    # print("TRAIN DF:", train_df["Date"].min(), "to", train_df["Date"].max())
    # print("VAL DF:", val_df["Date"].min(), "to", val_df["Date"].max())
    print("TEST DF:", test_df["Date"].min(), "to", test_df["Date"].max())
    # Remove Station6 entirely from training
    # dfs = {k: v for k, v in dfs.items() if k != test_station_name}

    return train_dfs, val_dfs, test_df




# Methods for visualizing the data splits, using the --visualize flag in main

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

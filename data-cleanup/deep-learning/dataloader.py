import math
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler


class MaskedDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for handling masked data.

    This dataset:
    - Takes in windowed data and feature names.
    - Applies random masking to simulate missing values.
    - Returns both the original and masked data, as well as the mask itself.

    Args:
        data (torch.Tensor): The input data tensor with shape (num_samples, window_size, num_features).
        features (List[str]): List of feature names corresponding to the columns in the data tensor.
        mask_amt (float): The proportion of values to mask in the data (default is 0.5).
    """
    
    def __init__(self, data, features, mask_amt=0.5):
        self.data = data
        self.mask_amt = mask_amt
        self.features = features
        
        # Indices of features to keep for model input (removes wind direction/speed)
        self.keep_indices = [
            i for i, f in enumerate(features)
            if f not in ["Wind direction", "Wind speed"]
        ]
        
        # Names of features used as input
        self.final_features = [features[i] for i in self.keep_indices]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary with:
            - 'original': original window (all features)
            - 'input': input window (features for model input)
            - 'masked': masked input window (same as input, but with masked values zeroed)
            - 'mask': boolean mask (True where value is masked)
        """
        
        # Get the original window
        original_window = self.data[idx]

        # Create a mask for this window (shape: window_size x num_features)
        mask = apply_mask(original_window.unsqueeze(0), self.features, self.mask_amt)[0]

        # Create masked version by zeroing out masked values
        masked_window = original_window.clone()
        masked_window[mask] = 0.0

        # Select only the features used for model input
        input_window = original_window[:, self.keep_indices]
        masked_input = masked_window[:, self.keep_indices]
        mask_input = mask[:, self.keep_indices]
        
        return {
            'original': original_window,      # All features, unmasked
            'input': input_window,            # Model input features, unmasked
            'masked': masked_input,           # Model input features, masked
            'mask': mask_input                # Mask for model input features
        }
    
# FEATURE ENGINEERING FUNCTIONS

# def add_precipitation_features(df) -> pd.DataFrame:
#     """
#     Add precipitation-related features
#     """
#     # 3-hour rolling sum
#     ppt_3h = df['Ppt'].rolling(3, min_periods=1)
#     df['Ppt_3h_sum'] = ppt_3h.sum()
#     mask = ppt_3h.apply(lambda x: x.isna().any(), raw=False)
#     df.loc[mask.astype(bool), 'Ppt_3h_sum'] = np.nan

#     # 24-hour rolling sum
#     ppt_24h = df['Ppt'].rolling(24, min_periods=1)
#     df['Ppt_24h_sum'] = ppt_24h.sum()
#     mask = ppt_24h.apply(lambda x: x.isna().any(), raw=False)
#     df.loc[mask.astype(bool), 'Ppt_24h_sum'] = np.nan

#     return df

def add_wind_features(df, station_id) -> pd.DataFrame:
    """
    Adds wind feature components (Wx, Wy) to the DataFrame based on wind speed and direction.
    This function computes the Cartesian components of wind (Wx, Wy) from the polar representation
    (wind speed and wind direction) and adds them as new columns to the input DataFrame. If either
    'Wind speed' or 'Wind direction' columns are missing, the function skips feature addition and
    prints a message indicating the missing columns for the given station.
    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'Wind speed' and 'Wind direction' columns.
        station_id (Any): Identifier for the station, used in logging if columns are missing.
    Returns:
        pd.DataFrame: DataFrame with added 'Wx' and 'Wy' columns if applicable.
    """
    if 'Wind speed' in df and 'Wind direction' in df:
        wv = df['Wind speed']
        wd_rad = np.deg2rad(df['Wind direction']) 
        df['Wx'] = wv * np.cos(wd_rad)
        df['Wy'] = wv * np.sin(wd_rad)
    else:
        print(f"Skipping wind features for Station {station_id} (columns missing)")
    
    return df

def add_time_features(df) -> pd.DataFrame:
    """
    Adds cyclical time-based features (sine and cosine transformations) to a DataFrame based on its datetime index.
    This function computes the sine and cosine transformations of the timestamp for daily, monthly, and yearly periods,
    and appends these as new columns to the input DataFrame. The new columns are named as 'DaySin', 'DayCos', 'MonthSin',
    'MonthCos', 'YearSin', and 'YearCos'.
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime-like index.
    Returns:
        pd.DataFrame: DataFrame with additional cyclical time feature columns.
    """
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    
    # Define time periods in seconds
    day = 86400
    year = 86400 * 365.2425
    month = 86400 * 30.4167
    
    # Add cyclical features
    time_features = {
        'Day': day,
        'Year': year,
        'Month': month
    }
    
    for period_name, period in time_features.items():
        df[f'{period_name}Sin'] = np.sin(timestamp_s * (2 * np.pi / period))
        df[f'{period_name}Cos'] = np.cos(timestamp_s * (2 * np.pi / period))
    
    return df

# def add_spatial_features(df, station_id) -> pd.DataFrame:
#     """
#     Add spatial features
#     """
#     loc_dict = {
#         1: (30.3989, -98.6105), 
#         2: (30.4193, -98.8046), 
#         3: (30.4421, -98.8427), 
#         4: (30.4600, -98.9407), 
#         5: (30.2454, -98.7059), 
#         6: (30.2758, -98.7242)
#     }
    
#     # df['latitude'], df['longitude'] = loc_dict[station_id]
#     # df['station_id'] = station_id
    
#     return df

def engineer_features(df, station_id) -> pd.DataFrame:
    """
    Perform feature engineering on the input DataFrame for a given station.

    This function:
    - Adds wind features (Wx, Wy) based on wind speed and direction.
    - (Optionally) Adds precipitation features (currently commented out).
    - Adds cyclical time features (sine/cosine for day, month, year).
    - Drops the 'Flag' column if present.
    - (Optionally) Adds spatial features (currently commented out).

    Args:
        df (pd.DataFrame): Input DataFrame with raw features.
        station_id (Any): Station identifier (used for logging).

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    print(f"Engineering features for station {station_id}")
    
    # Add wind features (Wx, Wy)
    df = add_wind_features(df, station_id)
    
    # Optionally add precipitation features
    # df = add_precipitation_features(df)
    
    # Add cyclical time features
    df = add_time_features(df)
    
    # Drop 'Flag' column
    df.drop('Flag', axis=1, inplace=True)
    
    # Optionally add spatial features
    # df = add_spatial_features(df, station_id)

    return df

# DATA NORMALIZATION AND WINDOWING FUNCTIONS
def normalize_features(df, features, lat_long_stats) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize selected features in the DataFrame using StandardScaler.

    This function:
    - Applies sklearn's StandardScaler to the specified feature columns.
    - (Optionally) Normalizes latitude and longitude columns (currently commented out).

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (List[str]): List of feature column names to normalize.
        lat_long_stats (tuple): (Optional) Tuple of (lat_mean, long_mean, lat_std, long_std) for spatial normalization.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Tuple of (normalized DataFrame, fitted scaler).
    """
    scaler = StandardScaler()
    
    df[features] = scaler.fit_transform(df[features])
    
    # Optionally normalize latitude and longitude (currently commented out)
    # lat_mean, long_mean, lat_std, long_std = lat_long_stats
    # df['latitude'] = (df['latitude'] - lat_mean) / lat_std
    # df['longitude'] = (df['longitude'] - long_mean) / long_std
        
    return df, scaler

# def get_lat_long_stats() -> Tuple[float, float, float, float]:
#     lat_sum, long_sum, lat_sq_sum, long_sq_sum = 0, 0, 0, 0
#     for station_id in range(1, 7):
#         base_dir = Path("../data-cleanup/feature_engineered")
#         file_path = base_dir / f"Station{station_id}_engineered_data.csv"

#         df_temp = pd.read_csv(file_path)
        
#         lat_sum += df_temp['latitude'][0]
#         long_sum += df_temp['longitude'][0]
#         lat_sq_sum += df_temp['latitude'][0]**2
#         long_sq_sum += df_temp['longitude'][0]**2
        
#     lat_mean, long_mean = lat_sum/6, long_sum/6
#     lat_std, long_std = lat_sq_sum / 6 - lat_mean**2, long_sq_sum / 6 - long_mean**2
    
#     return (lat_mean, long_mean, lat_std, long_std)

def make_windows(df, window_size=24, overlap = 0.5) -> torch.Tensor:
    """
    Splits a DataFrame into overlapping windows for time series modeling.

    Args:
        df (pd.DataFrame): Input DataFrame (no datetime column).
        window_size (int): Number of time steps per window.
        overlap (float): Fractional overlap between windows (0 <= overlap < 1).

    Returns:
        torch.Tensor: Tensor of shape (num_windows, num_features, window_size).
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 and 1")
    if len(df) == 0:
        raise ValueError("df cannot be empty")
    
    # df_numeric = df.drop('datetime', axis=1)
    # data = df_numeric.values
    data = df.values

    step_size = int(window_size * (1 - overlap))
    step_size = int(window_size * (1 - overlap))
    if step_size < 1:
        step_size = 1  # Ensure at least step of 1
    
    # Create sliding windows along the time axis
    windows = sliding_window_view(data, window_size, axis=0)[::step_size]
    
    # Filter out windows containing any NaNs
    valid_windows = ~np.isnan(windows).any(axis=(1, 2))
    
    # Return as (num_windows, num_features, window_size) for PyTorch
    return torch.tensor(windows[valid_windows], dtype=torch.float32).transpose(1, 2)
        

def load_station_data(station_id) -> pd.DataFrame:
    """
    Loads cleaned data for a given station from CSV.

    Args:
        station_id (int): Station identifier.

    Returns:
        pd.DataFrame: DataFrame with datetime index.
    """
    base_dir = Path("../cleaned_data")
    file_path = base_dir / f"Station{station_id}_cleaned_data.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def get_feature_columns(df, exclude_cols) -> List[str]:
    """
    Returns a list of feature columns to use for normalization/modeling.

    Excludes columns in exclude_cols and any columns containing 'Sin' or 'Cos'.

    Args:
        df (pd.DataFrame): Input DataFrame.
        exclude_cols (List[str]): Columns to exclude.

    Returns:
        List[str]: List of feature column names.
    """
    return [col for col in df.columns 
            if col not in exclude_cols 
            and ('Sin' not in col and 'Cos' not in col)]
    
def process_station_data(station_id, window_size, overlap, exclude_cols) -> Tuple[torch.Tensor, StandardScaler, List[str], List[str]]:
    """
    Loads, engineers, normalizes, and windows data for a single station.

    Args:
        station_id (int): Station identifier.
        window_size (int): Number of time steps per window.
        overlap (float): Fractional overlap between windows.
        exclude_cols (List[str]): Columns to exclude from normalization.

    Returns:
        Tuple containing:
            - windows (torch.Tensor): (num_windows, num_features, window_size)
            - scaler (StandardScaler): Fitted scaler for normalization
            - all_features (List[str]): All feature names in the DataFrame
            - norm_features (List[str]): Features used for normalization
    """
    # Load data
    df = load_station_data(station_id)
    
    # Feature engineering
    df = engineer_features(df, station_id)
    df = df.reset_index().rename(columns={'index': 'datetime'})
    df = df.drop('datetime', axis=1)
    
    # Get features and normalize
    all_features = list(df.columns)
    norm_features = get_feature_columns(df, exclude_cols)
    print(f"Station {station_id} - Normalizing {len(norm_features)} features")
    
    df, scaler = normalize_features(df, norm_features, None)
    
    # Create windows
    windows = make_windows(df, window_size, overlap)
    
    print(f"Processed data for Station {station_id}")
    
    return windows, scaler, all_features, norm_features

def get_windows_data(stations=None, window_size=24, overlap=0.5) -> Tuple[torch.Tensor, List[StandardScaler], List[str], List[int]]:
    """
    Loads, processes, and windows data for one or more stations.

    Args:
        stations (List[int] or None): List of station IDs to process. If None, processes all (1-6).
        window_size (int): Number of time steps per window.
        overlap (float): Fractional overlap between windows.

    Returns:
        Tuple containing:
            - windows_data (torch.Tensor): All windows concatenated (num_windows, num_features, window_size)
            - scalers (List[StandardScaler]): List of fitted scalers for each station
            - all_features (List[str]): List of all feature names (from first station)
            - norm_features (List[str]): List of normalized feature names (from first station)
    """
    if stations is None:
        stations = list(range(1, 7))

    exclude_cols = ['latitude', 'longitude', 'Ppt_RainFlag', 
                    'Ppt_3h_sum', 'Ppt_24h_sum', 'HoursSinceRain', 
                    "Wind direction", "Wind speed"]
    
    all_station_windows = []
    scalers = []
    
    all_features = None
    norm_feat_indices = None

    # lat_long_stats = get_lat_long_stats()
    for station_id in stations:  
        windows, scaler, station_all_features, station_norm_feat = process_station_data(
            station_id, window_size, overlap, exclude_cols
        )

        if all_features == None or norm_feat_indices == None:
            all_features = station_all_features
            norm_features = station_norm_feat
            
        all_station_windows.append(windows)
        scalers.append(scaler)
    
    # Concatenate all station windows along the batch dimension
    windows_data = torch.cat(all_station_windows, dim=0)

    return windows_data, scalers, all_features, norm_features


# MASKING FUNCTIONS

ORIGINAL_FEATURE_NUM = 14

def create_random_mask(data, mask_amt = 0.1) -> torch.BoolTensor:
    """
    Creates a random mask for the first ORIGINAL_FEATURE_NUM features in each window.

    Args:
        data (torch.Tensor): Shape (num_windows, window_size, num_features).
        mask_amt (float): Fraction of values to mask (0 < mask_amt <= 1).

    Returns:
        torch.BoolTensor: Mask of same shape as data, True where masked.
    """
    num_windows, window_size, _ = data.shape
    vals_per_window = window_size * ORIGINAL_FEATURE_NUM
    num_vals_mask = max(1, int(vals_per_window * mask_amt))
    
     # Generate random values for each value in the window
    rand = torch.rand((num_windows, vals_per_window))
    
     # Find the threshold for masking the lowest num_vals_mask values
    thresholds = torch.topk(rand, num_vals_mask, largest=False, dim=1).values[:, -1].unsqueeze(1)
    
    # Mask is True where the random value is below the threshold
    mask_flat = rand <= thresholds
    
    # Reshape to (num_windows, window_size, ORIGINAL_FEATURE_NUM)
    actual_mask = mask_flat.view(num_windows, window_size, ORIGINAL_FEATURE_NUM)
    
    # Place mask in the correct slice of the output mask
    mask = torch.zeros_like(data, dtype=torch.bool)
    mask[:,:window_size,:ORIGINAL_FEATURE_NUM] = actual_mask
    
    return mask.bool()

def create_block_mask(data, mask_amt = 0.1, max_block_size = 0,
                      force_full_row=False, force_full_col=False) -> torch.BoolTensor:
    """
    Creates a mask with random rectangular blocks for the first ORIGINAL_FEATURE_NUM features.

    Args:
        data (torch.Tensor): Shape (num_windows, window_size, num_features).
        mask_amt (float): Fraction of values to mask.
        max_block_size (int): Maximum block size (0 means no limit).

    Returns:
        torch.BoolTensor: Mask of same shape as data, True where masked.
    """
    
    num_windows, row_range, _ = data.shape
    col_range = ORIGINAL_FEATURE_NUM
    vals_per_window = row_range * col_range
    num_vals_mask = max(1, int(vals_per_window * mask_amt))
    
    mask = torch.zeros_like(data, dtype=torch.bool)
    
    for i in range(num_windows):
        masked_vals = 0
        while num_vals_mask > masked_vals:
            remaining_vals = num_vals_mask - masked_vals
            max_possible_block = min(remaining_vals, row_range * col_range)
            
            if max_block_size > 0:
                max_possible_block = min(max_possible_block, max_block_size)
            
            # Determine block dimensions
            if force_full_row:
                block_width = torch.randint(1, min(math.ceil(max_possible_block / col_range), row_range) + 1, (1,)).item()
                block_len = col_range
            elif force_full_col:
                block_width = row_range
                block_len = torch.randint(1, min(math.ceil(max_possible_block / row_range), col_range) + 1, (1,)).item()
            else:
                block_width = torch.randint(1, min(max_possible_block, row_range) + 1, (1,)).item()
                block_len = torch.randint(1, min(max_possible_block // block_width, col_range) + 1, (1,)).item()
                
            # Randomly choose block position
            r = torch.randint(0, row_range - block_width + 1, (1,)).item()
            c = torch.randint(0, col_range - block_len + 1, (1,)).item()
            
            masked_before = mask[i, r:r+block_width, c:c+block_len].sum().item()
            mask[i, r:r+block_width, c:c+block_len] = True       
            newly_masked = block_width * block_len - masked_before
            masked_vals += newly_masked

    return mask.bool()

def create_row_mask(data, mask_amt = 0.1):
    """
    Masks random full rows (timesteps) for the first ORIGINAL_FEATURE_NUM features.
    """
    return create_block_mask(data, mask_amt, force_full_row=True) 

def create_col_mask(data, mask_amt = 0.1):
    """
    Masks random full columns (features) for the first ORIGINAL_FEATURE_NUM features.
    """
    return create_block_mask(data, mask_amt, force_full_col=True)

#! Add later if needed
# def apply_ppt_mask(mask, base_mask, feature_list, dependents):
#     for dep_feat in dependents:
#         if dep_feat not in feature_list:
#             continue
#         dep_idx = feature_list.index(dep_feat)
#         mask[:, :, dep_idx]
        
#     return mask

def apply_wind_mask(mask, base_mask, feature_list, dependents):    
    dep_idxs = [feature_list.index(f) for f in dependents if f in feature_list]
    if not dep_idxs:
        return mask
    
    mask[:, :, dep_idxs] |= base_mask[:, :, None]
    return mask

#! Hard Coded for now
def apply_dependent_masking(mask, feature_list):
    """
    Enforces masking dependencies:
    If a base feature is masked at any timestep, its dependent features are also masked.
    """
    dependents = ['Wx', 'Wy']
    base_idx = feature_list.index('Wind direction')
    base_idx_2 = feature_list.index('Wind speed')
    base_mask = mask[:, :, base_idx] | mask[:, :, base_idx_2]

    mask = apply_wind_mask(mask, base_mask, feature_list, dependents)

    return mask

def apply_mask(data, features, tot_mask_amt = 0.5):
    mask_amts = np.random.dirichlet(np.ones(4)) * tot_mask_amt
    
    rand_mask = create_random_mask(data, mask_amts[0])
    block_mask = create_block_mask(data, mask_amts[1])
    row_mask = create_row_mask(data, mask_amts[2])
    col_mask = create_col_mask(data, mask_amts[3])
    
    combined_mask = rand_mask | block_mask | row_mask | col_mask
    
    final_mask = apply_dependent_masking(
        combined_mask, 
        features
    )
    return final_mask

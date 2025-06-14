import math
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler


class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, data, mask_amt=0.5):
        self.data = data
        self.mask_amt = mask_amt
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the original window
        original = self.data[idx]
        
        # Create mask for this window
        mask = apply_mask(original.unsqueeze(0), self.mask_amt)[0]
        
        # Create masked version by zeroing out masked values
        masked = original.clone()
        masked[mask] = 0.0
        
        return {
            'original': original,
            'masked': masked,
            'mask': mask
        }
    
    
def add_precipitation_features(df) -> pd.DataFrame:
    """
    Add precipitation-related features
    """
    # 3-hour rolling sum
    ppt_3h = df['Ppt'].rolling(3, min_periods=1)
    df['Ppt_3h_sum'] = ppt_3h.sum()
    mask = ppt_3h.apply(lambda x: x.isna().any(), raw=False)
    df.loc[mask.astype(bool), 'Ppt_3h_sum'] = np.nan

    # 24-hour rolling sum
    ppt_24h = df['Ppt'].rolling(24, min_periods=1)
    df['Ppt_24h_sum'] = ppt_24h.sum()
    mask = ppt_24h.apply(lambda x: x.isna().any(), raw=False)
    df.loc[mask.astype(bool), 'Ppt_24h_sum'] = np.nan
    
    return df

def add_wind_features(df, station_id) -> pd.DataFrame:
    """
    Add wind features
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
    Add cyclical time features
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
    print(f"Engineering features for station {station_id}")
    
    df = add_precipitation_features(df)
    df = add_wind_features(df, station_id)
    df = add_time_features(df)
    
    df.drop('Flag', axis=1, inplace=True)
    
    # Spatial features
    # df = add_spatial_features(df, station_id)

    return df

# Normalize features
def normalize_features(df, features, lat_long_stats) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    
    df[features] = scaler.fit_transform(df[features])
    
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
    windows = sliding_window_view(data, window_size, axis=0)[::step_size]
    
    valid_windows = ~np.isnan(windows).any(axis=(1, 2))
    
    return torch.tensor(windows[valid_windows], dtype=torch.float32).transpose(1, 2)
        

def load_station_data(station_id) -> pd.DataFrame:
    """
    Loads "clean" data for station
    """
    base_dir = Path("../cleaned_data")
    file_path = base_dir / f"Station{station_id}_cleaned_data.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def get_feature_columns(df, exclude_cols) -> List[str]:
    """
    Get the list of feature columns to use
    """
    return [col for col in df.columns 
            if col not in exclude_cols 
            and ('Sin' not in col and 'Cos' not in col)]
    
def process_station_data(station_id, window_size, overlap, exclude_cols) -> Tuple[torch.Tensor, StandardScaler, List[str], List[str]]:
    """
    Feature engineer and create window for station
    """
    # Load data
    df = load_station_data(station_id)
    
    # Feature engineering
    df = engineer_features(df, station_id)
    df = df.reset_index().rename(columns={'index': 'datetime'})
    df = df.drop('datetime', axis=1)
    
    # Get features and normalize
    all_features = df.columns
    norm_features = get_feature_columns(df, exclude_cols)
    norm_feat_indices = [df.columns.get_loc(feat) for feat in norm_features]
    
    df, scaler = normalize_features(df, norm_features, None)
    
    # Create windows
    windows = make_windows(df, window_size, overlap)
    
    print(f"Processed data for Station {station_id}")
    
    return windows, scaler, all_features, norm_feat_indices

def get_windows_data(stations=None, window_size=24, overlap=0.5) -> Tuple[torch.Tensor, List[StandardScaler], List[str], List[str]]:
    
    if stations is None:
        stations = list(range(1, 7))

    exclude_cols = ['latitude', 'longitude', 'Ppt_RainFlag', 
                    'Ppt_3h_sum', 'Ppt_24h_sum', 'HoursSinceRain', 
                    "Wind direction"]
    
    all_station_windows = []
    scalers = []
    
    all_features = None
    norm_feat_indices = None

    # lat_long_stats = get_lat_long_stats()
    for station_id in stations:  
        windows, scaler, station_all_features, station_norm_feat_indices = process_station_data(
            station_id, window_size, overlap, exclude_cols
        )

        if all_features == None or norm_feat_indices == None:
            all_features = station_all_features
            norm_feat_indices = station_norm_feat_indices
            
        all_station_windows.append(windows)
        scalers.append(scaler)
    
    windows_data = torch.cat(all_station_windows, dim=0)

    return windows_data, scalers, all_features, norm_feat_indices


ORIGINAL_FEATURE_NUM = 20

def create_random_mask(data, mask_amt = 0.1):
    
    num_windows, window_size, num_features = data.shape
    vals_per_window = window_size * ORIGINAL_FEATURE_NUM
    # print(vals_per_window)
    num_vals_mask = max(1, int(vals_per_window * mask_amt))
    
    # Produce a random distribution in the shape of (num_windows, values per window)
    rand = torch.rand((num_windows, vals_per_window))
    
    # Select the smallest 'num_vals_mask' random values and grabs the largest one as the threshold
    thresholds = torch.topk(rand, num_vals_mask, largest=False, dim=1).values[:, -1].unsqueeze(1)
    
    # Makes a flat mask where True are the 'num_vals_mask' random values that are below the threshold
    mask_flat = rand <= thresholds
    
    # Reshape mask to original data shape
    mask = torch.zeros_like(data, dtype=torch.bool)
    
    actual_mask = mask_flat.view(num_windows, window_size, ORIGINAL_FEATURE_NUM)
    
    mask[:,:window_size,:ORIGINAL_FEATURE_NUM] = actual_mask
    
    return mask.bool()

def create_block_mask(data, mask_amt = 0.1, max_block_size = 0):
    
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
            
            block_width = torch.randint(1, min(max_possible_block, row_range) + 1, (1,)).item()
            block_len = torch.randint(1, min(max_possible_block // block_width, col_range) + 1, (1,)).item()
                
            r = torch.randint(0, row_range - block_width + 1, (1,)).item()
            c = torch.randint(0, col_range - block_len + 1, (1,)).item()
            
            masked_before = mask[i, r:r+block_width, c:c+block_len].sum().item()
            
            mask[i, r:r+block_width, c:c+block_len] = True       
            
            newly_masked = block_width * block_len - masked_before
            masked_vals += newly_masked

    return mask.bool()

def create_row_mask(data, mask_amt = 0.1):
    
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
            
            block_len = col_range
            block_width = torch.randint(1, min(math.ceil(max_possible_block / block_len), row_range) + 1, (1,)).item()
            
            r = torch.randint(0, row_range - block_width + 1, (1,)).item()

            masked_before = mask[i, r:r+block_width,:col_range].sum().item()

            mask[i, r: r + block_width,:col_range] = True   
            
            newly_masked = block_width * block_len - masked_before
            masked_vals += newly_masked
        
        
    return mask.bool()

def create_col_mask(data, mask_amt = 0.1):
  
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
            
            block_width = row_range
            block_len = torch.randint(1, min(math.ceil(max_possible_block / block_width), col_range) + 1, (1,)).item()
            
            c = torch.randint(0, col_range - block_len + 1, (1,)).item()

            masked_before = mask[i,:, c:c+block_len].sum().item()
            
            mask[i,:, c:c+block_len] = True   
            
            newly_masked = block_width * block_len - masked_before
            masked_vals += newly_masked

    return mask.bool()

def apply_mask(data, tot_mask_amt = 0.5):
    mask_amts = np.random.dirichlet(np.ones(4)) * tot_mask_amt
    
    rand_mask = create_random_mask(data, mask_amts[0])
    block_mask = create_block_mask(data, mask_amts[1])
    row_mask = create_row_mask(data, mask_amts[2])
    col_mask = create_col_mask(data, mask_amts[3])
    
    final_mask = rand_mask | block_mask | row_mask | col_mask
    
    return final_mask

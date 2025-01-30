import numpy as np
import pandas as pd
import sklearn
from constants import TIME_COLUMNS, CYCLIC_COLUMNS

def add_time_features(df):
    """
    Adds time-based features such as hour, day, month, and cyclical encoding.
    """
    df = df.copy()
    # Extract time-based components from DatetimeIndex

    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    # Add cyclical features for day and month
    # Ensures that Dec 31st is close to Jan 1st in the cyclical feature space
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


from sklearn.preprocessing import MinMaxScaler

def normalize_features(df, target_col):
    """
    Scales data between 0 and 1 for LSTM training.
    """
    scaler = MinMaxScaler()
    df[target_col] = scaler.fit_transform(df[[target_col]])
    return df, scaler  # Return the fitted scaler to inverse transform later

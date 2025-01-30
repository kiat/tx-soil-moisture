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


from sklearn.preprocessing import MinMaxScaler

def normalize_features(train_df, test_df, target_col):
    """
    Scales training and test data separately to prevent data leakage.
    Ensures proper dtype conversion to avoid FutureWarnings.
    """
    scaler = MinMaxScaler()

    # Fit scaler on training data
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Flatten transformed values before assigning
    train_df.loc[:, target_col] = scaler.fit_transform(train_df[[target_col]]).flatten().astype(float)
    test_df.loc[:, target_col] = scaler.transform(test_df[[target_col]]).flatten().astype(float)

    return train_df, test_df, scaler  # Return the fitted scaler for inverse transformation

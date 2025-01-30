import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from constants import TIME_COLUMNS, CYCLIC_COLUMNS

### Utility function for cyclical encoding
def cyclical_encoding(df, column, max_val):
    """
    Adds sine and cosine transformations for a given time-based column.
    """
    df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / max_val)
    df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / max_val)
    return df

### Time Feature Engineering
def add_time_features(df):
    """
    Adds time-based cyclical features and drops the original raw time columns.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index

    # Extract raw time components
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week
    df["day_of_year"] = df.index.dayofyear

    # Apply cyclical encoding
    for col, max_val in zip(["hour", "day", "month", "week", "day_of_year"], [24, 31, 12, 52, 365]):
        df = cyclical_encoding(df, col, max_val)

    # Drop original time features (they are now encoded)
    return df.drop(columns=["hour", "day", "month", "week", "day_of_year"], errors="ignore")


### Fourier Seasonal Feature Engineering
def add_fourier_terms(df, num_terms=3):
    """
    Adds Fourier transform components to capture seasonality.
    """
    N = len(df)
    t = np.arange(N)

    for i in range(1, num_terms + 1):
        df[f"fourier_sin_{i}"] = np.sin(2 * np.pi * i * t / N)
        df[f"fourier_cos_{i}"] = np.cos(2 * np.pi * i * t / N)

    return df

### Feature Scaling
def normalize_features(train_df, test_df):
    """
    Scales all features in training and test data separately.
    Prevents data leakage by fitting only on train data.
    """
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)
    
    return train_scaled, test_scaled, scaler  # Return fitted scaler for inverse transform


def drop_columns(df, target_col):
    """
    Drops irrelevant features like extra `SWC_X` moisture columns (except the target),
    as well as static location-based features (Latitude, Longitude).
    """
    cols_to_drop = [col for col in df.columns if ("SWC" in col and col != target_col)]  # Remove other `SWC_X`
    cols_to_drop += ["Latitude", "Longitude"]  # Remove static location-based features

    return df.drop(columns=cols_to_drop, errors="ignore")  # `errors="ignore"` prevents crashes if column is missing

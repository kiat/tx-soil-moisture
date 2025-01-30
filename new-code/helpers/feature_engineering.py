import numpy as np
import pandas as pd
from constants import TIME_COLUMNS, CYCLIC_COLUMNS

def add_time_features(df):
    """
    Adds time-based features such as hour, day, month, and cyclical encoding.
    """
    df = df.copy()
    # Extract time-based components from DatetimeIndex
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

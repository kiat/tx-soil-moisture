# src/helpers/data_splitter.py
import pandas as pd
from helpers.data_helpers import data_to_X_y

def split_by_year(df, target_col, train_years, val_years, test_years, window_size=72, offset=24):
    """
    Splits the dataset by year instead of arbitrary percentages.

    Args:
        df (pd.DataFrame): The full dataset (must have a datetime index).
        target_col (str): The target variable (e.g., 'SWC_5').
        train_years (list): List of years to use for training.
        val_years (list): List of years to use for validation.
        test_years (list): List of years to use for testing.
        window_size (int): Number of time steps in each training example.
        offset (int): Forecasting offset.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Ensure index is a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The dataframe index must be a datetime index.")

    # Extract the year from the timestamp index
    df["Year"] = df.index.year

    # Split data based on specified years
    train_df = df[df["Year"].isin(train_years)].drop(columns=["Year"])
    val_df = df[df["Year"].isin(val_years)].drop(columns=["Year"])
    test_df = df[df["Year"].isin(test_years)].drop(columns=["Year"])

    # Convert to supervised learning format
    X_train, y_train = data_to_X_y(train_df[target_col], window_size, offset)
    X_val, y_val = data_to_X_y(val_df[target_col], window_size, offset)
    X_test, y_test = data_to_X_y(test_df[target_col], window_size, offset)

    print(f"Data split by year: Train {train_years}, Val {val_years}, Test {test_years}")
    print(f"Shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def leave_one_station_one_year_out(df_dict, target_col, test_station, test_year, window_size=72, offset=24):
    """
    Trains on all stations and years except for one specific station-year (used for testing).

    Args:
        df_dict (dict): Dictionary where keys are station names and values are DataFrames.
        target_col (str): The target variable (e.g., 'SWC_5').
        test_station (str): The station to exclude from training (used for testing).
        test_year (int): The year to exclude from training (used for testing).
        window_size (int): Number of time steps in each training example.
        offset (int): Forecasting offset.

    Returns:
        X_train, y_train, X_test, y_test
    """
    train_dfs = []
    test_df = None

    for station, df in df_dict.items():
        df["Year"] = df.index.year  # Ensure year is extracted from datetime index

        if station == test_station:
            test_df = df[df["Year"] == test_year].drop(columns=["Year"])
        else:
            train_dfs.append(df[df["Year"] != test_year].drop(columns=["Year"]))

    if test_df is None or len(train_dfs) == 0:
        raise ValueError("Test station or test year not found in the dataset.")

    # Combine all training data into one DataFrame
    train_df = pd.concat(train_dfs)

    # Convert to supervised learning format
    X_train, y_train = data_to_X_y(train_df[target_col], window_size, offset)
    X_test, y_test = data_to_X_y(test_df[target_col], window_size, offset)

    print(f"Leave-One-Station-One-Year-Out Split: Train (excluding {test_station}-{test_year})")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def leave_one_station_one_year_out(df_dict, target_col, test_station, test_year, window_size=72, offset=24):
    """
    Trains on all stations and years except for one specific station-year (used for testing).

    Args:
        df_dict (dict): Dictionary where keys are station names and values are DataFrames.
        target_col (str): The target variable (e.g., 'SWC_5').
        test_station (str): The station to exclude from training (used for testing).
        test_year (int): The year to exclude from training (used for testing).
        window_size (int): Number of time steps in each training example.
        offset (int): Forecasting offset.

    Returns:
        X_train, y_train, X_test, y_test
    """
    train_dfs = []
    test_df = None

    for station, df in df_dict.items():
        df["Year"] = df.index.year  # Ensure year is extracted from datetime index

        if station == test_station:
            test_df = df[df["Year"] == test_year].drop(columns=["Year"])
        else:
            train_dfs.append(df[df["Year"] != test_year].drop(columns=["Year"]))

    if test_df is None or len(train_dfs) == 0:
        raise ValueError("Test station or test year not found in the dataset.")

    # Combine all training data into one DataFrame
    train_df = pd.concat(train_dfs)

    # Convert to supervised learning format
    X_train, y_train = data_to_X_y(train_df[target_col], window_size, offset)
    X_test, y_test = data_to_X_y(test_df[target_col], window_size, offset)

    print(f"Leave-One-Station-One-Year-Out Split: Train (excluding {test_station}-{test_year})")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, y_train, X_test, y_test
"""Data loading and preprocessing utilities."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def prepare_dataloaders(
    stations,
    target_station,
    window_size,
    offset,
    batch_size,
    predictors=None,
    predict_features=None,
    predict_avg=False,
    label_width=1,
    training_stride=1,
    validation_stride=1,
):
    """
    Load, engineer, split, normalize data and create PyTorch DataLoaders.

    Returns: (train_loader, val_loader, test_loader, all_features, input_dim, data_shape)
        where data_shape is a dict with keys: 'batch_size', 'time_steps', 'num_features'
    """
    # Import or define placeholder data helpers
    try:
        from core.data_helpers import (
            read_and_process_csvs,
            engineer_features,
            split_and_stack_data,
            normalize_features,
            data_to_X_y,
        )
    except ImportError:
        print("Warning: 'core' module not found. Using placeholder data functions.")

        def read_and_process_csvs():
            return {
                f"Station{i+1}": pd.DataFrame(
                    np.random.rand(100, 6),
                    columns=["SWC_20", "T_20", "Ppt", "Tair", "Wx", "Wy"],
                )
                for i in range(6)
            }

        def engineer_features(dfs):
            return dfs

        def split_and_stack_data(dfs, test_station_name):
            return dfs, dfs[test_station_name], dfs[test_station_name]

        def normalize_features(df, features):
            return df, None

        def data_to_X_y(df, window, offset, ):
            return np.random.rand(50, window, len(df.columns)), np.random.rand(50)

    # --- Feature Setup ---
    # Set to predict daily averages if specified
    labels_list = predict_features.split(",") if predict_features else ["SWC_20"]
    if predict_avg:
        labels_list = [f"{label}_daily_avg" for label in labels_list]

    predictors = (
        predictors.split(",")
        if predictors
        else ["SWC_20", "T_20", "Ppt", "Tair", "Wx", "Wy"]
    )

    # Combine all features for Xy data preparation
    all_features = np.concatenate([predictors, labels_list])
    indices = [int(np.where(all_features == f)[0][0]) for f in labels_list]

    # Load and process data
    raw_dfs = read_and_process_csvs()
    engineered_dfs = engineer_features(raw_dfs, daily_average=predict_avg, predict_features=predict_features.split(","))
    train_dfs, val_dfs, test_df = split_and_stack_data(
        engineered_dfs, test_station_name=target_station
    )

    # Prepare test set
    scaled_test, _ = normalize_features(test_df, all_features)
    X_test, y_test = data_to_X_y(scaled_test, window_size, offset, label_width, indices)
    train_data_x, train_data_y = [], []

    # Prepare training set (all stations except target)
    for df in train_dfs:
        scaled_train, _ = normalize_features(df, all_features)
        X_train_part, y_train_part = data_to_X_y(scaled_train, window_size, offset, label_width, indices)
        
        X_train_part = X_train_part[::training_stride]
        y_train_part = y_train_part[::training_stride]
        
        train_data_x.append(X_train_part)
        train_data_y.append(y_train_part)
    X_train = np.concatenate(train_data_x, axis=0)
    y_train = np.concatenate(train_data_y, axis=0)

    # Prepare validation set
    val_data_x, val_data_y = [], []
    for df in val_dfs:
        scaled, _ = normalize_features(df, all_features)
        X_part, y_part = data_to_X_y(scaled, window_size, offset, label_width, indices)
        X_part = X_part[::validation_stride]
        y_part = y_part[::validation_stride]
        val_data_x.append(X_part)
        val_data_y.append(y_part)
    X_val = np.concatenate(val_data_x, axis=0)
    y_val = np.concatenate(val_data_y, axis=0)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)

    input_dim = len(all_features)

    # Data shape info for special models (e.g., ILSTM_Soil)
    data_shape = {
        "batch_size": X_train_t.shape[0],
        "time_steps": X_train_t.shape[1],
        "num_features": X_train_t.shape[2],
    }

    return train_loader, val_loader, test_loader, all_features, input_dim, data_shape


def get_output_helpers():
    """Get or define evaluation output helper functions."""
    try:
        from core.evaluation_helpers import (
            write_loss_history_to_csv,
            write_model_results_to_csv,
        )

        return write_loss_history_to_csv, write_model_results_to_csv
    except ImportError:

        def write_loss_history_to_csv(*args):
            pass

        def write_model_results_to_csv(station, model_name, ws, offset, perf, features, labels):
            print(f"DUMMY_WRITE: Writing results for {model_name}...")
            print(
                f"  -> MSE: {perf.get('MSE', 'N/A')}, CORR: {perf.get('CORR', 'N/A')}"
            )

        return write_loss_history_to_csv, write_model_results_to_csv

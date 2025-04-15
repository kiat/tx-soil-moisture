import argparse
import os
import csv
import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, SimpleRNN, Conv1D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from scipy.stats import pearsonr

from data_helpers import *
import models as model_module

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt

import models as model_module
from model_entry import ModelEntry

from utils import get_model_id, format_model_name



##########################################
def smape(y_true, y_pred, epsilon=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = 2 * np.abs(y_pred - y_true) / np.maximum(denominator, epsilon)
    return np.mean(smape) * 100


##########################################
def compute_rse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.sqrt(numerator / denominator) if denominator != 0 else np.nan


###############################################################################

def evaluate_model(model, X_test, y_test):
    """
    Evaluation of Models. Metrics are: MSE, MAE, MAPE, SMAPE, RSE, CORR
    """
    predictions = model.predict(X_test).flatten()
    y_test = y_test.flatten()
    
    # Ensure no shape mismatches
    if predictions.shape != y_test.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, y_test {y_test.shape}")

    final_results =  {
        "mean_squared_error": mean_squared_error(y_test, predictions),
        "mean_absolute_error":mean_absolute_error(y_test , predictions),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y_test, predictions),
        "smape": smape(y_test, predictions),
        "rse": compute_rse(y_test, predictions),
        "corr": pearsonr(y_test, predictions).statistic
    }

    print(final_results)
    return final_results

###############################################################################



def write_loss_history_to_csv(station, model_name, window_size, offset, history, feature_str):
    """Saves loss history to a unique CSV file including offset and feature set."""
    
     # Ensure the results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
 
    # Define loss history file path inside the results folder
    loss_file = os.path.join(results_dir, f"loss_history_ws{window_size}_offset{offset}_{feature_str}.csv")
    
    # Check if the file already exists
    file_exists = os.path.isfile(loss_file)
    
    # Define CSV headers
    headers = ["Station", "Model", "Features", "Offset", "Epoch", "Loss", "Validation Loss"]

    # Open in write mode if file exists (reset each run)
    mode = "a" if file_exists else "w"

    with open(loss_file, mode=mode, newline="") as file:
        writer = csv.writer(file)
        
        # Write headers only if file is new
        if not file_exists:
            writer.writerow(headers)  

        # Write training history
        for epoch, (loss, val_loss) in enumerate(zip(history["loss"], history["val_loss"])):
            writer.writerow([station, model_name, feature_str, offset, epoch + 1, loss, val_loss])

    print(f"Saved loss history for {model_name} (ws={window_size}, offset={offset}, features={feature_str}) on {station} to {loss_file}")

###############################################################################
###############################################################################

    
def write_model_results_to_csv(station, model_name, window_size, offset, performance, feature_str):

    # Ensure the results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"results_ws{window_size}_offset{offset}_{feature_str}.csv")

    file_exists = os.path.isfile(results_file)
    headers = ["Station", "Model", "Features", "Offset", "MSE", "MAE", "MAPE", "SMAPE", "RSE", "CORR"]
    
    # Open in write mode if file exists (reset each run)
    mode = "a" if file_exists else "w"
    
    with open(results_file, mode=mode, newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            station, model_name, feature_str, offset,
            performance.get("mean_squared_error"),
            performance.get("mean_absolute_error"),
            performance.get("mean_absolute_percentage_error"),
            performance.get("smape"),
            performance.get("rse"),
            performance.get("corr")
        ])
    print(f"Saved model results for {model_name} on {station} with {len(feature_str.split('_'))} features to {results_file}")


###############################################################################
###############################################################################


    
def main(args):
    
    

    # engineer_and_save_data()
    stations = ['Station1', 'Station2', 'Station3', 'Station4', 'Station5', 'Station6']
    target_station = stations[-1]  # Dynamically select the target station

    # Load and process raw CSVs in-memory
    raw_dfs = read_and_process_csvs()
    engineered_dfs = engineer_features(raw_dfs)
    
    
    
    # Split data:
    # - `val_df` → Past years of target station (validation)
    # - `test_df` → Current year of target station (testing)
    # - `train_dfs` → All other stations (training)
    engineered_dfs, val_df, test_df = split_and_stack_data(engineered_dfs, test_station_name=target_station, remove_met=False)


    if 0:
        print("FEATURES IN THE DATA\n")
        for station, df in engineered_dfs.items():
            print(f"--- {station} ---")
            print(df.describe())  # Summary statistics
            print("\n")
    all_features = args.features.split(',') if args.features else ['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']

    # Visualize the split if and only if the --visualize flag is set

    if args.visualize:
        # Prepare unscaled Date-indexed DataFrames
        val_df_plot = val_df.set_index("Date")
        test_df_plot = test_df.set_index("Date")
        train_df_plot = concatenate_with_gaps([
            df.set_index("Date") for name, df in engineered_dfs.items()
            if name != target_station
        ])


        # Choose a feature to visualize
        feature_for_plot = all_features[0]
        plot_split_timeline(train_df_plot, val_df_plot, test_df_plot, feature=feature_for_plot)

        return  # Exit before training

    

    # Prepare validation & test sets
    scaled_val, _ = normalize_features(val_df, all_features)
    X_val, y_val = data_to_X_y(scaled_val, args.window_size, args.offset)

    scaled_test, _ = normalize_features(test_df, all_features)
    X_test, y_test = data_to_X_y(scaled_test, args.window_size, args.offset)

    # Prepare training data: merge all stations except target station
    train_data = []  
    for train_station in [s for s in stations if s != target_station]:
        print(f"Adding {train_station} to training pool...")
        scaled_train, _ = normalize_features(engineered_dfs[train_station], all_features)
        X_train, y_train = data_to_X_y(scaled_train, args.window_size, args.offset)
        train_data.append((X_train, y_train))

    # Merge all training data into a single dataset
    X_train = np.concatenate([data[0] for data in train_data], axis=0)
    y_train = np.concatenate([data[1] for data in train_data], axis=0)

    print(f"\nTraining on {len(stations)-1} stations and testing on {target_station}...\n")

    # Ensure y is the right shape
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Print for debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Features being used: {all_features}")


    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Normalize model IDs from CLI input like: "lstm,attention_lstm"
    def normalize_id(name: str) -> str:
        return name.lower().replace("_", "")

    requested_ids = set(normalize_id(n) for n in args.model_names.split(","))

    # Build models dynamically from compile_* functions in models.py
    models_we_process = {}

    for name in dir(model_module):
        if name.startswith("compile_"):
            internal_name = name.replace("compile_", "")        # e.g. attention_lstm
            model_id = normalize_id(internal_name)              # e.g. attentionlstm

            if model_id in requested_ids:
                compile_fn = getattr(model_module, name)
                models_we_process[model_id] = ModelEntry(internal_name, compile_fn)
    print(f"Models to be processed: {models_we_process.keys()}")
    # Train each model with transfer learning
    for model_id, model_entry in models_we_process.items():
        model = model_entry.build((args.window_size, len(all_features)))

        print(f"\nTraining {model_entry} across stations...\n")  # __str__ used automatically

        if model_id == "baseline":
            model.fit(X_train, y_train)
            performance = evaluate_model(model, X_test, y_test)

            note_path = os.path.join(model_dir, f"model_{model_id}_NOTE.txt")
            with open(note_path, "w") as f:
                f.write("Baseline model - no weights saved.\n")

            feature_str = '_'.join(all_features)
            write_model_results_to_csv(target_station, str(model_entry), args.window_size, args.offset, performance, feature_str)
            print(f"{model_entry} Final Test Loss: {performance['mean_squared_error']}\n")
            continue

        # Compile and train model
        model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=0.001),
            metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)]
        )

        print("\nFinal Evaluation on Test Set...\n")
        performance = evaluate_model(model, X_test, y_test)

        # Save trained model
        feature_str = '_'.join(all_features)
        main_name = f"model_{model_id}_ws{args.window_size}_offset{args.offset}_{feature_str}"
        model_path = os.path.join(model_dir, f"{main_name}.keras")
        model.save(model_path)
        print(f"{model_entry} saved at {model_path}")

        # Save results
        write_model_results_to_csv(target_station, str(model_entry), args.window_size, args.offset, performance, feature_str)
        write_loss_history_to_csv(target_station, str(model_entry), args.window_size, args.offset, history.history, feature_str)

        print(f"{model_entry} Final Test Loss: {performance['mean_squared_error']}\n")

    print("All Runs Complete! All results saved.")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train various models for time series prediction.")
    parser.add_argument('--window_size', type=int, default=168, help='Window size for input data')
    parser.add_argument('--offset', type=int, default=24, help='Offset for prediction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument("--features", type=str, default="SWC_20,T_20,Ppt,Tair,Wx,Wy", help="Comma-separated list of features to use in training")
    parser.add_argument("--model_names", type=str, default="LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline",  help="Comma-separated list of Models short form like LSTM,CNN")

    # If this is set, no models will be trained, but the train/val/test splits will be visualized
    # and saved to the results folder.
    parser.add_argument('--visualize', action='store_true', help='If true, plots train/val/test splits instead of running models')

    args = parser.parse_args()
    main(args)

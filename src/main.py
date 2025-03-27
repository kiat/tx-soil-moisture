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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from scipy.stats import pearsonr

from preprocess_data import read_and_process_csvs, engineer_features
from models import * 

import numpy as np


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



##########################################
def normalize_data(df, features):
    scaler = MinMaxScaler()

    no_scale_features = [feat for feat in features if 'sin' in feat or 'cos' in feat]
    scale_features = [feat for feat in features if feat not in no_scale_features]

    df = df.reset_index(drop=True)

    scaled_data = scaler.fit_transform(df[scale_features])
    scaled_df = pd.DataFrame(scaled_data, columns=scale_features)

    scaled_df = pd.concat([scaled_df, df[no_scale_features]], axis=1)

    scaled_df = scaled_df[features]

    return scaled_df.to_numpy(), scaler

###########################################

# def data_to_X_y(data, window_size, offset):
#     X, y = [], []
#     for i in range(len(data) - window_size - offset):
#         X.append(data[i:i+window_size, :])  
#         y.append(data[i + window_size + offset, 0])  

#     return  np.array(X),  np.array(y)


def data_to_X_y(data, window_size, offset):
    rows = len(data) - window_size - offset
    X = np.lib.stride_tricks.sliding_window_view(data, (window_size, data.shape[1]))[:rows, 0]
    y = data[window_size + offset : window_size + offset + rows, 0]
    return X, y

###############################################################################
###############################################################################
###############################################################################
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
###############################################################################
###############################################################################
###############################################################################

    
def split_and_stack_data(dfs, test_station_name="Station6", remove_met=False):
    if remove_met:
        for key in dfs.keys():
            dfs[key] = dfs[key][["SWC_5", "SWC_10", "SWC_20", "SWC_50"]]

    test_df = dfs[test_station_name].loc['2020-01-01 00:00:00':]
    val_df = dfs[test_station_name].loc[:'2020-12-31 23:00:00']
    
    dfs[test_station_name] = dfs[test_station_name].drop(test_df.index)
    
    return dfs, val_df, test_df
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
###############################################################################
###############################################################################


    
def main(args):

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
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
    print("FEATURES IN THE DATA\n")
    for station, df in engineered_dfs.items():
        print(f"--- {station} ---")
        print(df.describe())  # Summary statistics
        print("\n")
    all_features = args.features.split(',') if args.features else ['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']

    # Prepare validation & test sets
    scaled_val, _ = normalize_data(val_df, all_features)
    X_val, y_val = data_to_X_y(scaled_val, args.window_size, args.offset)

    scaled_test, _ = normalize_data(test_df, all_features)
    X_test, y_test = data_to_X_y(scaled_test, args.window_size, args.offset)

    # Prepare training data: merge all stations except target station
    train_data = []  
    for train_station in [s for s in stations if s != target_station]:
        print(f"Adding {train_station} to training pool...")
        scaled_train, _ = normalize_data(engineered_dfs[train_station], all_features)
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

    # Define models to train
    models = {
        "LSTM": compile_lstm((args.window_size, len(all_features))),
        "BiLSTM": compile_bilstm((args.window_size, len(all_features))),
        "RNN": compile_rnn((args.window_size, len(all_features))),
        "CNN": compile_cnn((args.window_size, len(all_features))),
        "AttentionLSTM": compile_attention_lstm((args.window_size, len(all_features))),
        "Autoregressive": compile_autoregressive((args.window_size, len(all_features))),
        "Baseline": Baseline(),
    }

    # Keep only the models that are passed by arguments.
    model_names = set(args.model_names.split(","))
    models_we_process = {key: models[key] for key in model_names if key in models}

    # Train each model with transfer learning
    for model_name, model in models_we_process.items():
        print(f"\n Training {model_name} across stations...\n")
        
        if model_name == "Baseline":
            model.fit(X_train, y_train)
            performance = evaluate_model(model, X_test, y_test)

            model_path = os.path.join(model_dir, f"model_{model_name}_NOTE.txt")
            with open(model_path, "w") as f:
                f.write("Baseline model - no weights saved.\n")

            write_model_results_to_csv(target_station, model_name, args.window_size, args.offset, performance, '_'.join(all_features))
            print(f"{model_name} Final Test Loss: {performance['mean_squared_error']}\n")
            continue  # skip to next model
        
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()])
        
        # Train on all stations EXCEPT the target station
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),  # Validate on past years of target station
            epochs=args.epochs,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)]
        )

        print("\nFinal Evaluation on Test Set...\n")
        performance = evaluate_model(model, X_test, y_test)  

        # Save the trained model
        # model_path = os.path.join("models", f"{model_name}.keras")

        main_name = f"model_{model_name}_ws{args.window_size}_offset{args.offset}_{args.features}"
        model_path = os.path.join(model_dir, f"{main_name}.keras")
        model.save(model_path)
        print(f"{model_name} saved at {model_path}")

        # Save results
        write_model_results_to_csv(target_station, model_name, args.window_size, args.offset, performance, '_'.join(all_features))
        write_loss_history_to_csv(target_station, model_name, args.window_size, args.offset, history.history, '_'.join(all_features))

        print(f"{model_name} Final Test Loss: {performance['mean_squared_error']}\n")

    print("All Runs Complete! All results saved.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train various models for time series prediction.")
    parser.add_argument('--window_size', type=int, default=168, help='Window size for input data')
    parser.add_argument('--offset', type=int, default=24, help='Offset for prediction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument("--features", type=str, default="SWC_20,T_20,Ppt,Tair,Wx,Wy", help="Comma-separated list of features to use in training")
    parser.add_argument("--model_names", type=str, default="LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline",  help="Comma-separated list of Models short form like LSTM,CNN")
    args = parser.parse_args()
    main(args)

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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

def smape(y_true, y_pred):
    numerator = 2 * tf.abs(y_pred - y_true)
    denominator = tf.abs(y_true) + tf.abs(y_pred) + tf.keras.backend.epsilon()
    return 100 * tf.reduce_mean(numerator / denominator)

def compute_rse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.sqrt(numerator / denominator) if denominator != 0 else np.nan

def compute_corr(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    denominator = np.sqrt(np.sum((y_true - y_true_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2))
    return numerator / denominator if denominator != 0 else np.nan



def normalize_data(df, features):
    scaler = MinMaxScaler()
    data = df[features]
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def data_to_X_y(data, window_size, offset):
    X, y = [], []
    for i in range(len(data) - window_size - offset):
        X.append(data[i:i+window_size, :])  
        y.append(data[i + window_size + offset, 0])  
    return np.array(X), np.array(y)

def compile_lstm(input_shape, learning_rate=0.0001):
    model = Sequential([
        LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        Bidirectional(LSTM(32, return_sequences=False)), 
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate), metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError(), smape])
    return model

def compile_bilstm(input_shape, learning_rate=0.0001):
    model = Sequential([
        Bidirectional(LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(32, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate), metrics=[RootMeanSquaredError(), smape])
    return model


def compile_rnn(input_shape, learning_rate=0.0001):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        SimpleRNN(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate), metrics=[RootMeanSquaredError(), smape])
    return model

def compile_cnn(input_shape, learning_rate=0.0001):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate), metrics=[RootMeanSquaredError()])
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()
    return {
        "r2_score": r2_score(y_test, predictions),
        "mean_squared_error": mean_squared_error(y_test, predictions),
        "mean_absolute_error": np.mean(np.abs(y_test - predictions)),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y_test, predictions),
        "smape": np.mean(2 * np.abs(predictions - y_test) / (np.abs(y_test) + np.abs(predictions) + 1e-8)) * 100,
        "rse": compute_rse(y_test, predictions),
        "corr": compute_corr(y_test, predictions)
    }
    
def split_and_stack_data(dfs, test_station_name="Station6", remove_met=False):
    if remove_met:
        for key in dfs.keys():
            dfs[key] = dfs[key][["SWC_5", "SWC_10", "SWC_20", "SWC_50"]]

    test_df = dfs[test_station_name].loc['2020-01-01 00:00:00':]
    val_df = dfs[test_station_name].loc[:'2020-12-31 23:00:00']
    
    dfs[test_station_name] = dfs[test_station_name].drop(test_df.index)
    
    return dfs, val_df, test_df

def write_loss_history_to_csv(station, model_name, window_size, offset, history, feature_str):
    """Saves loss history to a unique CSV file including offset and feature set."""
    
    # Define unique filename per run
    loss_file = f"loss_history_{model_name}_ws{window_size}_offset{offset}_{feature_str}.csv"
    
    # Check if the file already exists
    file_exists = os.path.isfile(loss_file)
    
    # Define CSV headers
    headers = ["Station", "Model", "Features", "Offset", "Epoch", "Loss", "Validation Loss"]

    # Open in write mode if file exists (reset each run)
    mode = "w" if file_exists else "a"

    with open(loss_file, mode, newline="") as file:
        writer = csv.writer(file)
        
        # Write headers only if file is new
        if not file_exists:
            writer.writerow(headers)  

        # Write training history
        for epoch, (loss, val_loss) in enumerate(zip(history["loss"], history["val_loss"])):
            writer.writerow([station, model_name, feature_str, offset, epoch + 1, loss, val_loss])

    print(f"Saved loss history for {model_name} (ws={window_size}, offset={offset}, features={feature_str}) on {station} to {loss_file}")
    
    
def write_model_results_to_csv(station, model_name, window_size, offset, performance, feature_str):
    results_file = f"results_{model_name}_ws{window_size}_offset{offset}_{feature_str}.csv"
    file_exists = os.path.isfile(results_file)
    headers = ["Station", "Model", "Features", "Offset", "R2", "MSE", "MAE", "MAPE", "SMAPE", "RSE", "CORR"]
    
    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            station, model_name, feature_str, offset,
            performance.get("r2_score"),
            performance.get("mean_squared_error"),
            performance.get("mean_absolute_error"),
            performance.get("mean_absolute_percentage_error"),
            performance.get("smape"),
            performance.get("rse"),
            performance.get("corr")
        ])
    print(f"Saved model results for {model_name} on {station} with {len(feature_str.split('_'))} features to {results_file}")

from preprocess_data import read_and_save_parquet, engineer_and_save_data

    
def main(args):
    # read_and_save_parquet()
    # engineer_and_save_data()
    stations = ['Station1', 'Station2', 'Station3', 'Station4', 'Station5', 'Station6']
    target_station = stations[-1]  # Dynamically select the target station

    # Load all station data
    engineered_dfs = {station: pd.read_parquet(f"{station}_engineered.parquet") for station in stations}

    # Split data:
    # - `val_df` → Past years of target station (validation)
    # - `test_df` → Current year of target station (testing)
    # - `train_dfs` → All other stations (training)
    engineered_dfs, val_df, test_df = split_and_stack_data(engineered_dfs, test_station_name=target_station, remove_met=False)

    all_features = args.features.split(',') if args.features else ['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']

    # Prepare validation & test sets
    scaled_val, _ = normalize_data(val_df, all_features)
    X_val, y_val = data_to_X_y(scaled_val, args.window_size, args.offset)

    scaled_test, _ = normalize_data(test_df, all_features)
    X_test, y_test = data_to_X_y(scaled_test, args.window_size, args.offset)

    # Prepare training data: merge all stations except target station
    train_data = []  
    for train_station in [s for s in stations if s != target_station]:
        print(f"✅ Adding {train_station} to training pool...")
        scaled_train, _ = normalize_data(engineered_dfs[train_station], all_features)
        X_train, y_train = data_to_X_y(scaled_train, args.window_size, args.offset)
        train_data.append((X_train, y_train))

    # Merge all training data into a single dataset
    X_train = np.concatenate([data[0] for data in train_data], axis=0)
    y_train = np.concatenate([data[1] for data in train_data], axis=0)

    print(f"\n🔹 Training on {len(stations)-1} stations and testing on {target_station}...\n")

    # Define models to train
    models = {
        "LSTM": compile_lstm((args.window_size, len(all_features)), learning_rate=0.0001),
        "BiLSTM": compile_bilstm((args.window_size, len(all_features)), learning_rate=0.0001),
        "RNN": compile_rnn((args.window_size, len(all_features)), learning_rate=0.0001),
        "CNN": compile_cnn((args.window_size, len(all_features)), learning_rate=0.0001)
    }

    # Train each model with transfer learning
    for model_name, model in models.items():
        print(f"\n🔹 Training {model_name} across stations...\n")

        # Train on all stations EXCEPT the target station
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),  # Validate on past years of target station
            epochs=args.epochs,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)]
        )

        print("\n🔹 Final Evaluation on Test Set...\n")
        performance = evaluate_model(model, X_test, y_test)  

        # Save the trained model
        model_path = os.path.join("models", f"{model_name}.keras")
        model.save(model_path)
        print(f"✅ {model_name} saved at {model_path}")

        # Save results
        write_model_results_to_csv(target_station, model_name, args.window_size, args.offset, performance, '_'.join(all_features))
        write_loss_history_to_csv(target_station, model_name, args.window_size, args.offset, history.history, '_'.join(all_features))

        print(f"✅ {model_name} Final Test Loss: {performance['mean_squared_error']}\n")

    print("✅ All Runs Complete! All results saved.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train various models for time series prediction.")
    parser.add_argument('--window_size', type=int, default=168, help='Window size for input data')
    parser.add_argument('--offset', type=int, default=24, help='Offset for prediction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument("--features", type=str, default="SWC_20,T_20,Ppt,Tair,Wx,Wy",
                    help="Comma-separated list of features to use in training")

    args = parser.parse_args()
    main(args)

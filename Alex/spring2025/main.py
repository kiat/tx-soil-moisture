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
        "smape": np.mean(2 * np.abs(predictions - y_test) / (np.abs(y_test) + np.abs(predictions) + 1e-8)) * 100
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
    """Saves model evaluation results to a uniquely named CSV file including offset and feature set."""
    
    # Define unique filename per run
    results_file = f"results_{model_name}_ws{window_size}_offset{offset}_{feature_str}.csv"
    
    # Check if file already exists
    file_exists = os.path.isfile(results_file)

    # Define headers
    headers = ["Station", "Model", "Features", "Offset", "R2", "MSE", "MAE", "MAPE", "SMAPE"]

    # Extract metrics safely
    mse = performance.get("mean_squared_error", None)
    mae = performance.get("mean_absolute_error", None)
    mape = performance.get("mean_absolute_percentage_error", None)
    smape_score = performance.get("smape", None)
    r2 = performance.get("r2_score", None)

    # Write to CSV
    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write headers if it's a new file
        if not file_exists:
            writer.writerow(headers)

        # Append results
        writer.writerow([station, model_name, feature_str, offset, r2, mse, mae, mape, smape_score])

    print(f"Saved model results for {model_name} (ws={window_size}, offset={offset}, features={feature_str}) on {station} to {results_file}")


def main(args):
    stations = ['Station1', 'Station2', 'Station3', 'Station4', 'Station5', 'Station6']
    engineered_dfs = {station: pd.read_parquet(f"{station}_engineered.parquet") for station in stations}

    engineered_dfs, val_df, test_df = split_and_stack_data(engineered_dfs, test_station_name="Station6", remove_met=False)

    # all_features = ['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']
    all_features = args.features.split(',') if args.features else ['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']
    
    
    models = {
        "LSTM": compile_lstm((args.window_size, 1), learning_rate=0.0001),
        "BiLSTM": compile_bilstm((args.window_size, 1), learning_rate=0.0001),
        "RNN": compile_rnn((args.window_size, 1), learning_rate=0.0001),
        "CNN": compile_cnn((args.window_size, 1), learning_rate=0.0001)
    }

    history_dicts = {}
    performance = {}
    val_performance = {}

    for i in range(1, len(all_features) + 1):
        selected_features = all_features[:i]  # Incrementally add features
        print(f"\nTraining models with features: {selected_features}")

        for train_station in [s for s in stations if s != "Station6"]:
            print(f"\nTraining models on {train_station}...")

            scaled_train, scaler = normalize_data(engineered_dfs[train_station], selected_features)
            scaled_val, _ = normalize_data(val_df, selected_features)
            scaled_test, _ = normalize_data(test_df, selected_features)

            X_train, y_train = data_to_X_y(scaled_train, args.window_size, args.offset)
            X_val, y_val = data_to_X_y(scaled_val, args.window_size, args.offset)
            X_test, y_test = data_to_X_y(scaled_test, args.window_size, args.offset)

            for name, model in models.items():
                print(f"\nTraining {name} model on {train_station} with {len(selected_features)} features...")

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=args.epochs,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)]
                )

                # Save history
                history_dicts[f"{name}_{train_station}_{'_'.join(selected_features)}"] = history.history

                # Evaluate model
                performance[name] = evaluate_model(model, X_test, y_test)
                val_performance[name] = evaluate_model(model, X_val, y_val)

                # Save model
                feature_str = '_'.join(selected_features)
                model_path = os.path.join("models", f"{name}_{train_station}_{feature_str}.keras")
                model.save(model_path)
                print(f"{name} model saved at {model_path}")

                # Save results with feature names in the filename
                write_model_results_to_csv(train_station, name, args.window_size, args.offset, performance[name], feature_str)
                write_loss_history_to_csv(train_station, name, args.window_size, args.offset, history.history, feature_str)

                print(f"{name} Model Test Loss with {len(selected_features)} features: {performance[name]['mean_squared_error']}")

    print("Training complete! All results saved.")

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

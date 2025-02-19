import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

def smape(y_true, y_pred):
    numerator = 2 * tf.abs(y_pred - y_true)
    denominator = tf.abs(y_true) + tf.abs(y_pred) + tf.keras.backend.epsilon()  # Avoid division by zero
    return 100 * tf.reduce_mean(numerator / denominator)

def normalize_data(df, features):
    """Scales selected features using MinMaxScaler."""
    scaler = MinMaxScaler()
    data = df[features]
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def data_to_X_y(data, window_size, offset):
    """Converts time series data into LSTM-ready sequences."""
    X, y = [], []
    for i in range(len(data) - window_size - offset):
        X.append(data[i:i+window_size, :])  # Multiple features
        y.append(data[i + window_size + offset, 0])  # Predicting first feature
    return np.array(X), np.array(y)

def compile_model(input_shape, learning_rate=0.0001):
    """Defines and compiles an LSTM model."""
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate), metrics=[RootMeanSquaredError()])
    return model

def fit_model(X_train, y_train, X_val, y_val, model_path, epochs, patience):
    """Trains an LSTM model with early stopping and checkpoint saving."""
    model = compile_model((X_train.shape[1], X_train.shape[2]), learning_rate=0.0001)
    
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks
    )
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and returns performance metrics."""
    predictions = model.predict(X_test).flatten()
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    return predictions, r2, mse, mape

def split_and_stack_data(dfs, test_station_name="Station6", remove_met=False):
    """Splits training, validation, and test data per station."""
    if remove_met:
        for key in dfs.keys():
            dfs[key] = dfs[key][["SWC_5", "SWC_10", "SWC_20", "SWC_50"]]

    test_df = dfs[test_station_name].loc['2020-01-01 00:00:00':]  # Test set from 2020+
    val_df = dfs[test_station_name].loc[:'2020-12-31 23:00:00']  # Validation set ≤2020
    
    # Remove test data from training to avoid leakage
    dfs[test_station_name] = dfs[test_station_name].drop(test_df.index)
    
    return dfs, val_df, test_df

def main(args):
    # Load preprocessed data from Parquet files
    stations = ['Station1', 'Station2', 'Station3', 'Station4', 'Station5', 'Station6']
    engineered_dfs = {station: pd.read_parquet(f"{station}_engineered.parquet") for station in stations}

    # Split into training, validation, and test sets
    engineered_dfs, val_df, test_df = split_and_stack_data(engineered_dfs, test_station_name="Station6", remove_met=False)

    # Training on selected features
    features = ['SWC_20']

    # Define models to train
    models = {
        "LSTM": compile_model((args.window_size, len(features)), learning_rate=0.0001),
        # "GRU": compile_gru_model((args.window_size, len(features)), learning_rate=0.0001)  # Assuming you have a GRU model function
    }

    history_dicts = {}
    performance = {}
    val_performance = {}

    # Loop through each training station instead of stacking them
    for train_station in [s for s in stations if s != "Station6"]:
        print(f"\nTraining models on {train_station}...")

        # Normalize data for each station
        scaled_train, scaler = normalize_data(engineered_dfs[train_station], features)
        scaled_val, _ = normalize_data(val_df, features)
        scaled_test, _ = normalize_data(test_df, features)

        # Convert to LSTM-ready format
        X_train, y_train = data_to_X_y(scaled_train, args.window_size, args.offset)
        X_val, y_val = data_to_X_y(scaled_val, args.window_size, args.offset)
        X_test, y_test = data_to_X_y(scaled_test, args.window_size, args.offset)

        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name} model on {train_station}...")

            # Define EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)

            # Compile the model
            model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[
                    tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.MeanAbsolutePercentageError(),
                    smape
                ]
            )

            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                callbacks=[early_stopping]
            )

            # Save history
            config_name = f"{features} - {args.window_size} input / {args.offset} output"
            history_dicts[f"{name}_{train_station}"] = history.history

            # Evaluate model
            performance[name] = model.evaluate(X_test, y_test, return_dict=True)
            val_performance[name] = model.evaluate(X_val, y_val, return_dict=True)

            # Save model
            model_path = os.path.join("models", f"{name}_{train_station}.keras")
            model.save(model_path)
            print(f"{name} model saved at {model_path}")

            # Save results
            results_filename = os.path.join("results", f"{train_station}_results.csv")
            # write_model_results_to_csv(train_station, name, performance[name], filename=results_filename)
            # write_loss_history_to_csv(train_station, name, history.history, loss_file=results_filename)

            print(f"{name} Model Test Loss: {performance[name]}")

    print("Training complete! All results saved.")
    # Save all results to CSV
    results_df = pd.DataFrame(results, columns=['Station', 'Offset', 'R2', 'MSE', 'MAPE'])
    results_filename = 'results_all_stations.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"Saved all results to CSV: {results_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model for time series prediction.")
    parser.add_argument('--window_size', type=int, default=168, help='Window size for input data')
    parser.add_argument('--offset', type=int, default=24, help='Offset for prediction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')

    args = parser.parse_args()
    main(args)

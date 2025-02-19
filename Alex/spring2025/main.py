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

    # Split test & validation data from test station
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
    
    # Loop through each training station instead of stacking them
    results = []
    for train_station in [s for s in stations if s != "Station6"]:
        print(f"Training on {train_station}...")
        
        # Normalize data for each station
        scaled_train, scaler = normalize_data(engineered_dfs[train_station], features)
        scaled_val, _ = normalize_data(val_df, features)
        scaled_test, _ = normalize_data(test_df, features)
        
        # Convert to LSTM-ready format
        X_train, y_train = data_to_X_y(scaled_train, args.window_size, args.offset)
        X_val, y_val = data_to_X_y(scaled_val, args.window_size, args.offset)
        X_test, y_test = data_to_X_y(scaled_test, args.window_size, args.offset)
        
        # Train the model
        model_path = f"model_{train_station}.keras"
        model, history = fit_model(X_train, y_train, X_val, y_val, model_path, args.epochs, args.patience)
        
        # Evaluate on test data
        predictions, r2, mse, mape = evaluate_model(model, X_test, y_test)
        
        # Store results
        results.append([train_station, args.offset, r2, mse, mape])
        print(f'Finished training on {train_station} - R2 Score: {r2:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}')
        
        # Save trained model
        model.save(model_path)
        print(f"Trained model saved at: {model_path}")
    
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

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from itertools import combinations
import tensorflow as tf
import os
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_sequences(data, n_steps, offset=1):
    X, y = [], []
    for i in range(len(data) - n_steps - offset + 1):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps + offset - 1])
    return np.array(X), np.array(y)

def create_lstm_model(n_steps, n_features):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_autoregressive_lstm_model(n_steps, n_features):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_feature_combination(df, target_col, feature_cols, n_steps=7, offset=1, model_type='lstm'):
    # Prepare data
    data = df[feature_cols + [target_col]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Split into train and test
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, n_steps, offset)
    X_test, y_test = create_sequences(test_data, n_steps, offset)
    
    # Create and train model based on type
    if model_type == 'lstm':
        model = create_lstm_model(n_steps, len(feature_cols) + 1)
    else:  # autoregressive
        model = create_autoregressive_lstm_model(n_steps, len(feature_cols) + 1)
    
    model.fit(X_train, y_train[:, -1], epochs=50, batch_size=32, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_orig = scaler.inverse_transform(np.hstack([np.zeros((len(y_test), len(feature_cols))), y_test[:, -1].reshape(-1, 1)]))[:, -1]
    y_pred_orig = scaler.inverse_transform(np.hstack([np.zeros((len(y_pred), len(feature_cols))), y_pred]))[:, -1]
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    mae_percentage = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
    
    return mse, mae, mae_percentage

def run_analysis(stations=["Station1"], target="Sat_SM_AMSR", 
                model_type="lstm", n_steps=7):
    """
    Run the analysis with configurable parameters
    
    Args:
        stations (list): List of station names to analyze together (e.g., ["Station1", "Station2"])
                        Data from all specified stations will be combined for analysis
        target (str): Target column to analyze ("Sat_SM_AMSR" or "Sat_SM_SMAP")
        model_type (str): Model type to use ("lstm" or "autoregressive")
        n_steps (int): Number of steps to use for prediction
    """
    # Validate inputs
    valid_stations = [f"Station{i}" for i in range(1, 7)]
    valid_targets = ["Sat_SM_AMSR", "Sat_SM_SMAP"]
    valid_models = ["lstm", "autoregressive"]
    
    # Input validation
    if not all(station in valid_stations for station in stations):
        raise ValueError(f"Stations must be from: {valid_stations}")
    if target not in valid_targets:
        raise ValueError(f"Target must be one of: {valid_targets}")
    if model_type not in valid_models:
        raise ValueError(f"Model type must be one of: {valid_models}")
    
    # Create results directory if it doesn't exist
    results_dir = 'satellite/LSTM_results'
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    # Load and combine data from all specified stations
    combined_df = pd.DataFrame()
    for station in stations:
        df = pd.read_csv(f'satellite/met_merged_satelite_data/{station}_Met_AMSR_SMAP.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Station'] = station  # Add station identifier
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Set index and sort by date
    combined_df = combined_df.set_index(['Date', 'Station']).sort_index()
    
    # Define features for the target
    feature_cols = {
        'Sat_SM_AMSR': ['Ppt', 'Srad', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'distance_AMSR'],
        'Sat_SM_SMAP': ['Ppt', 'Srad', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'distance_SMAP']
    }
    
    # Remove rows with missing values
    combined_df = combined_df.dropna()
    
    # Calculate total combinations for progress tracking
    total_combinations = sum(len(list(combinations(feature_cols[target], r))) 
                           for r in range(1, len(feature_cols[target]) + 1))
    current = 0

    print(f"Testing combined data from {', '.join(stations)} - {target} with {model_type} model: "
          f"n_steps={n_steps} ({current}/{total_combinations} combinations tested)")
    
    for r in range(1, len(feature_cols[target]) + 1):
        for combo in combinations(feature_cols[target], r):
            mse, mae, mae_percentage = evaluate_feature_combination(
                combined_df, 
                target, 
                list(combo), 
                n_steps=n_steps,
                offset=n_steps,
                model_type=model_type
            )
            results.append({
                'stations': ', '.join(stations),
                'target': target,
                'model_type': model_type,
                'n_steps': n_steps,
                'features': combo,
                'mse': mse,
                'mae': mae,
                'mae_percentage': mae_percentage
            })
            current += 1

    # Convert results to DataFrame and sort by MAE percentage
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mae_percentage')

    # Print results
    print(f"\nBest results for combined data from {', '.join(stations)} - {target}:")
    print(results_df[
        ['model_type', 'n_steps', 'features', 'mse', 'mae', 'mae_percentage']
    ].head(10).to_string())

    # Generate descriptive filename
    config_str = f"{'-'.join(stations)}_{target}_{model_type}_steps{n_steps}"
    results_path = os.path.join(results_dir, f"{config_str}.csv")
    
    # Save results with configuration details at the top
    with open(results_path, 'w') as f:
        f.write(f"# Configuration:\n")
        f.write(f"# Stations: {', '.join(stations)}\n")
        f.write(f"# Target: {target}\n")
        f.write(f"# Model Type: {model_type}\n")
        f.write(f"# Steps: {n_steps}\n")
        f.write("#\n")
        results_df.to_csv(f, index=False)
    
    print(f"\nResults saved to: {results_path}")
    
    return results_df

# Example usage:
if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Run LSTM/Autoregressive analysis on satellite data')
    parser.add_argument('--stations', nargs='+', help='List of stations to analyze together')
    parser.add_argument('--target', type=str, help='Target variable (Sat_SM_AMSR or Sat_SM_SMAP)')
    parser.add_argument('--model_type', type=str, help='Model type (lstm or autoregressive)')
    parser.add_argument('--steps', type=int, help='Number of steps for prediction')
    
    args = parser.parse_args()
    
    # Use the parsed arguments
    results_df = run_analysis(
        stations=args.stations,
        target=args.target,
        model_type=args.model_type,
        n_steps=args.steps
    )

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from itertools import combinations
import tensorflow as tf
import os

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

# Load and prepare data
df = pd.read_csv('satellite/met_merged_satelite_data/Station1_Met_AMSR_SMAP.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Define features and targets
target_cols = ['Sat_SM_AMSR', 'Sat_SM_SMAP']
feature_cols = {
    'Sat_SM_AMSR': ['Ppt', 'Srad', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'distance_AMSR'],
    'Sat_SM_SMAP': ['Ppt', 'Srad', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'distance_SMAP']
}

# Remove rows with missing values
df = df.dropna()

# Test all combinations
results = []
total_combinations = 2 * 2 * 7 * sum(len(list(combinations(feature_cols['Sat_SM_AMSR'], r))) for r in range(1, len(feature_cols['Sat_SM_AMSR']) + 1))
current = 0

for target_col in target_cols:
    for model_type in ['lstm', 'autoregressive']:
        for steps in range(1, 8):  # steps will be used for both n_steps and offset
            print(f"Testing {target_col} with {model_type} model: n_steps=offset={steps} ({current}/{total_combinations} combinations tested)")
            for r in range(1, len(feature_cols[target_col]) + 1):
                for combo in combinations(feature_cols[target_col], r):
                    mse, mae, mae_percentage = evaluate_feature_combination(
                        df, 
                        target_col, 
                        list(combo), 
                        n_steps=steps,  # same value for both
                        offset=steps,    # same value for both
                        model_type=model_type
                    )
                    results.append({
                        'target': target_col,
                        'model_type': model_type,
                        'n_steps': steps,
                        'offset': steps,
                        'features': combo,
                        'mse': mse,
                        'mae': mae,
                        'mae_percentage': mae_percentage
                    })
                    current += 1

# Sort results by MSE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mse')

# Print results
print("\nTop 10 Combinations for each target:")
for target in target_cols:
    target_results = results_df[results_df['target'] == target]
    print(f"\nBest results for {target}:")
    print(target_results[['model_type', 'n_steps', 'offset', 'features', 'mse', 'mae', 'mae_percentage']].head(10).to_string())

# Save results
results_df.to_csv('satellite/results/model_comparison_results_both_targets.csv', index=False)

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

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

def create_model(n_steps, n_features):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_feature_combination(df, target_col, feature_cols, n_steps=7):
    # Prepare data
    data = df[feature_cols + [target_col]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Split into train and test
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, n_steps)
    X_test, y_test = create_sequences(test_data, n_steps)
    
    # Train model
    model = create_model(n_steps, len(feature_cols) + 1)
    model.fit(X_train, y_train[:, -1], epochs=50, batch_size=32, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform for actual metrics
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

# Define features and target
target_col = 'Sat_SM_SMAP'
feature_cols = ['Ppt', 'Srad', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'Sat_SM_AMSR']

# Remove rows with missing values
df = df.dropna()

# Test all feature combinations
results = []
for r in range(1, len(feature_cols) + 1):
    for combo in combinations(feature_cols, r):
        mse, mae, mae_percentage = evaluate_feature_combination(df, target_col, list(combo))
        results.append({
            'features': combo,
            'mse': mse,
            'mae': mae,
            'mae_percentage': mae_percentage
        })

# Sort results by MSE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mse')

# Print results
print("\nTop 10 Feature Combinations:")
print(results_df.head(10).to_string())

# Create results directory if it doesn't exist
os.makedirs('satellite/results', exist_ok=True)

# Save results
results_df.to_csv('satellite/results/feature_combinations.csv', index=False)

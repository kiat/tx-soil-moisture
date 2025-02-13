import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

def read_data():
    dfs = {}
    for index in range(6):
        path = f'Station{index + 1}_Revised_Final_Data.csv'
        print(f"Reading: {path}")
        df = pd.read_csv(path, sep=",")
        df.columns = df.columns.str.replace(' ', '')
        df.insert(0, 'Date', pd.to_datetime(df['Unnamed:0']))
        df.drop('Unnamed:0', axis=1, inplace=True)
        df.set_index('Date', inplace=True)
        
        for col in ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50', 'T_5', 'T_10', 'T_20', 'T_50', 'Ppt']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        dfs[f'Station{index + 1}'] = df
    return dfs

def engineer_data(df):
    wv = df.pop('Windspeed')
    wd_rad = df.pop('Winddirection') * np.pi / 180
    max_wv = np.max(wv)
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)
    
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    day, year, month = 24 * 60 * 60, 24 * 3600 * 365.2425, 24 * 60 * 60 * 30.4167
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    
    return df.reset_index(drop=True)

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

def compile_model(input_shape, learning_rate):
    model = Sequential([
        InputLayer(input_shape),
        LSTM(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate), metrics=[RootMeanSquaredError()])
    return model

def fit_model(X_train, y_train, X_val, y_val, model_path, epochs, patience):
    model = compile_model((X_train.shape[1], X_train.shape[2]), learning_rate=0.0001)
    callbacks = [ModelCheckpoint(model_path, save_best_only=True), EarlyStopping(monitor='val_loss', patience=patience)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=callbacks)
    return model, history

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    return predictions, r2, mse, mape

def plot_results(history, predictions, y_test):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(predictions, label='Predictions')
    plt.plot(y_test, label='Actual')
    plt.legend()
    plt.title('Predictions vs Actual')
    plt.show()

def main(args):
    dfs = read_data()
    df = engineer_data(dfs['Station1'])
    df_test = engineer_data(dfs['Station6'])
    features = ['SWC_20', 'T_20', 'Day cos', 'Ppt']
    test_data, _ = normalize_data(df_test, features)
    scaled_data, _ = normalize_data(df, features)
    
    X, y = data_to_X_y(scaled_data, args.window_size, args.offset)
    
    # getting test data from different station
    X_test, y_test = data_to_X_y(test_data, args.window_size, args.offset)
    
    n = len(X)
    X_train, y_train = X[:int(n*0.7)], y[:int(n*0.7)]
    X_val, y_val = X[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9)]
    # X_test, y_test = X[int(n*0.9):], y[int(n*0.9):]
    
    model, history = fit_model(X_train, y_train, X_val, y_val, args.model_path, args.epochs, args.patience)
    predictions, r2, mse, mape = evaluate_model(model, X_test, y_test)
    
    print(f'R2 Score: {r2:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}')
    
    
    results_filename = f'results_offset_{args.offset}.csv'
    results_df = pd.DataFrame([[args.offset, r2, mse, mape]], columns=['Offset', 'R2', 'MSE', 'MAPE'])
    results_df.to_csv(results_filename, index=False)
    
    # plot_results(history, predictions, y_test)

# 24633504 7038528 3519264
# (36657, 10474, 5237)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model for time series prediction.")
    parser.add_argument('--window_size', type=int, default=168, help='Window size for input data')
    parser.add_argument('--offset', type=int, default=24, help='Offset for prediction')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--model_path', type=str, default='model.keras', help='Path to save the trained model')
    parser.add_argument('--patience', type=int, default=3, help='patience for model. 3 is defaulted')
    
    args = parser.parse_args()
    main(args)

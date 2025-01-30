import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from constants import ARIMA_ORDER, WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE

# Train ARIMA Model
def train_arima(df, target_col):
    """
    Trains an ARIMA model on the given DataFrame.
    Returns the trained model and predictions.
    """
    train_size = int(len(df) * 0.8)
    train, test = df[target_col][:train_size], df[target_col][train_size:]

    model = ARIMA(train, order=ARIMA_ORDER)
    model_fit = model.fit()

    predictions = model_fit.forecast(steps=len(test))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test, label="Actual")
    plt.plot(test.index, predictions, label="Predicted", linestyle="dashed")
    plt.legend()
    plt.title(f"ARIMA Predictions for {target_col}")
    plt.show()

    return model_fit, predictions

# Prepare LSTM Data
def prepare_lstm_data(df, target_col):
    """
    Converts time-series data into sequences for LSTM training.
    """
    X, y = [], []
    data = df[target_col].values

    for i in range(len(data) - WINDOW_SIZE):
        X.append(data[i : i + WINDOW_SIZE])
        y.append(data[i + WINDOW_SIZE])

    return np.array(X).reshape(-1, WINDOW_SIZE, 1), np.array(y)

# Train LSTM Model
def train_lstm(df, target_col):
    """
    Trains an LSTM model on the given time-series data.
    """
    X_train, y_train = prepare_lstm_data(df, target_col)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(32, return_sequences=False),
        Dense(1)
    ])
    
    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE)

    return model

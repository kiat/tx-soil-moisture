import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from helpers.constants import ARIMA_ORDER, WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE

def train_arima(df, target_col):
    """
    Trains an ARIMA model on the given DataFrame.
    """
    train_size = int(len(df) * 0.8)
    train, test = df[target_col][:train_size], df[target_col][train_size:]

    model = ARIMA(train, order=ARIMA_ORDER)
    model_fit = model.fit()

    predictions = model_fit.forecast(steps=len(test))
    
    return model_fit, predictions

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

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from constants import WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE

from sklearn.model_selection import train_test_split

def prepare_lstm_data(train_df, test_df, target_col):
    """
    Converts time-series data into sequences for LSTM training.
    Splits into X_train, X_test, y_train, y_test.
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    
    train_data = train_df[target_col].values
    test_data = test_df[target_col].values

    for i in range(len(train_data) - WINDOW_SIZE):
        X_train.append(train_data[i : i + WINDOW_SIZE])
        y_train.append(train_data[i + WINDOW_SIZE])

    for i in range(len(test_data) - WINDOW_SIZE):
        X_test.append(test_data[i : i + WINDOW_SIZE])
        y_test.append(test_data[i + WINDOW_SIZE])

    return (
        np.array(X_train).reshape(-1, WINDOW_SIZE, 1), np.array(y_train),
        np.array(X_test).reshape(-1, WINDOW_SIZE, 1), np.array(y_test)
    )


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

def train_lstm(train_df, test_df, target_col):
    """
    Trains an LSTM model on the given time-series data.
    """
    X_train, y_train, X_test, y_test = prepare_lstm_data(train_df, test_df, target_col)

    from tensorflow.keras.layers import Bidirectional

    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True), input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    
    model.compile(loss="mse", optimizer="adam")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr]  # Add scheduler
    )
    print(f"LSTM model trained successfully on {target_col}.")
    return model



def make_lstm_predictions(model, df, target_col):
    """
    Uses the trained LSTM model to generate forecasts.
    """
    X_test = df[target_col].values[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)  # Use last WINDOW_SIZE values
    predictions = []
    
    for _ in range(24):  # Forecast next 24 hours
        pred = model.predict(X_test)[0][0]
        predictions.append(pred)
        X_test = np.roll(X_test, -1)  # Shift input window
        X_test[0, -1, 0] = pred  # Add new prediction

    return np.array(predictions)

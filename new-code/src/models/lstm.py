import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

def build_simple_lstm_model(window_size):
    """Builds a simple LSTM model."""
    model = Sequential([
        InputLayer(input_shape=(window_size, 1)),
        LSTM(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def build_optimized_lstm_model(window_size):
    """Builds an optimized LSTM model with bidirectional layers and dropout."""
    model = Sequential([
        InputLayer(input_shape=(window_size, 1)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

def build_cnn_model(window_size):
    """Builds a 1D CNN model for time series forecasting."""
    model = Sequential([
        InputLayer(input_shape=(window_size, 1)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

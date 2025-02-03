import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from constants import WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE, NUM_PREDICTIONS


def train_lstm(X_train, y_train, X_test, y_test, epochs = 10, batch_size = 32):
    """
    Trains an LSTM model using Bidirectional LSTM layers.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),  # Single LSTM layer
        Dense(1)  # Output layer
    ])
    
    model.compile(loss="mse", optimizer="adam")

    model.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test))
    
    print("LSTM model trained successfully.")
    return model


def make_lstm_predictions(model, df, target_col,num_predictions=NUM_PREDICTIONS):
    """
    Uses the trained LSTM model to generate rolling forecasts.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    all_cols = [target_col] + feature_cols
    
    # Ensure the DataFrame has all expected columns
    df = df.reindex(columns=all_cols, fill_value=0)

    # Convert to NumPy array with float32 type
    input_seq = df[all_cols].values[-WINDOW_SIZE:].astype(np.float32).reshape(1, WINDOW_SIZE, len(all_cols))

    predictions = []

    for _ in range(num_predictions):
        pred = model.predict(input_seq, verbose=0)[0][0]  # Suppress unnecessary logs
        predictions.append(pred)

        # Shift window forward correctly
        input_seq = np.roll(input_seq, -1, axis=1)
        input_seq[0, -1, :] = np.append(pred, input_seq[0, -1, 1:])  # Insert new prediction in the right place

    return np.array(predictions, dtype=np.float32)

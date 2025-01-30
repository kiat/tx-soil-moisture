import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from constants import WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE, NUM_PREDICTIONS

####################################
# 1. Prepare Data for LSTM
####################################
def prepare_lstm_data(train_df, test_df, target_col):
    """
    Converts time-series data into sequences for LSTM training.
    Ensures correct data types and handles NaNs.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train_df!")

    train_df, test_df = train_df.astype(np.float32), test_df.astype(np.float32)
    train_data, test_data = train_df.values, test_df.values
    target_index = train_df.columns.get_loc(target_col)

    if len(train_data) <= WINDOW_SIZE or len(test_data) <= WINDOW_SIZE:
        raise ValueError(f"Data too small for WINDOW_SIZE={WINDOW_SIZE}.")

    def create_sequences(data):
        X, y = [], []
        for i in range(len(data) - WINDOW_SIZE):
            X.append(data[i : i + WINDOW_SIZE])
            y.append(data[i + WINDOW_SIZE, target_index])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    return *create_sequences(train_data), *create_sequences(test_data)

####################################
# 2. Train LSTM Model
####################################
def train_lstm(X_train, y_train, X_test, y_test):
    """
    Trains an LSTM model using Bidirectional LSTM layers.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(256, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])
    
    model.compile(loss="mse", optimizer="adam")
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
                 EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

    model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, validation_data=(X_test, y_test), callbacks=callbacks)
    print("LSTM model trained successfully.")
    return model

####################################
# 3. Make Predictions
####################################
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

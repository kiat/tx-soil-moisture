import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, \
    Dropout, Bidirectional, BatchNormalization, EarlyStopping, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from constants import WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE


####################################
# 1. Prepare Data for LSTM
####################################
import numpy as np
from constants import WINDOW_SIZE

import numpy as np
from constants import WINDOW_SIZE

def prepare_lstm_data(train_df, test_df, target_col):
    """
    Converts time-series data into sequences for LSTM training.
    Assumes train_df and test_df have already been preprocessed.
    """
    # Ensure `target_col` exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train_df!")

    # Convert to NumPy for faster indexing
    train_data = train_df.values
    test_data = test_df.values

    # Identify target column index
    target_index = train_df.columns.get_loc(target_col)

    # Check dataset size before creating sequences
    if len(train_data) <= WINDOW_SIZE or len(test_data) <= WINDOW_SIZE:
        raise ValueError(f"Data is too small for WINDOW_SIZE={WINDOW_SIZE}. Increase data or decrease WINDOW_SIZE.")

    # Vectorized creation of LSTM sequences
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data) - WINDOW_SIZE):
            X.append(data[i : i + WINDOW_SIZE])
            y.append(data[i + WINDOW_SIZE, target_index])  # Ensure correct target selection
        return np.array(X), np.array(y)

    # Prepare train and test sequences
    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    return X_train, y_train, X_test, y_test

####################################
# 2. Train LSTM Model
####################################
def train_lstm(X_train, y_train, X_test, y_test):
    """
    Trains an LSTM model using Bidirectional LSTM layers.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)

    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(256, return_sequences=True, input_shape=input_shape)),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(32, activation="relu"),
        # Use linear activation if your target is scaled 0..1 or you just want raw regression output
        Dense(1, activation="linear")  
    ])

    model.compile(loss="mse", optimizer="adam")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stop]
    )

    print("LSTM model trained successfully.")
    return model

####################################
# 3. Make Predictions
####################################
def make_lstm_predictions(model, df, target_col):
    """
    Uses the trained LSTM model to generate rolling forecasts.
    Predicts future time steps iteratively.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    all_cols = [target_col] + feature_cols

    input_seq = df[all_cols].values[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, len(all_cols))
    predictions = []

    for _ in range(24):  # Forecast next 24 time steps
        pred = model.predict(input_seq)[0][0]
        predictions.append(pred)

        # Shift window forward
        input_seq = np.roll(input_seq, -1, axis=1)
        input_seq[0, -1, 0] = pred  # Append prediction as new input

    return np.array(predictions)

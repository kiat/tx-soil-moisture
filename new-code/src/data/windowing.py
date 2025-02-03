import numpy as np

def generate_windows(train_df, test_df, target_col, WINDOW_SIZE=24):
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
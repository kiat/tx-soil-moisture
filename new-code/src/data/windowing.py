import numpy as np

def data_to_X_y(series, window_size, offset):
    data = series.values
    X, y = [], []
    for i in range(len(data) - window_size - offset):
        X.append(data[i:i+window_size].reshape(-1, 1))
        y.append(data[i+window_size+offset])
    return np.array(X), np.array(y)

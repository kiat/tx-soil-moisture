import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_predictions(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    maep = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Error Percentage

    metrics = {
        "R² Score": r2,
        "Mean Squared Error (MSE)": mse,
        "Mean Absolute Error (MAE)": mae,
        "Mean Absolute Error Percentage (MAEP)": maep
    }

    return metrics

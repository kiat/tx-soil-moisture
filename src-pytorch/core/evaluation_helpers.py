
import os
import csv
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr

# This module contains evaluation functions for machine learning models.

# Calculate SMAPE
def smape(y_true, y_pred, epsilon=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = 2 * np.abs(y_pred - y_true) / np.maximum(denominator, epsilon)
    return np.mean(smape) * 100


# Calculate RSE
def compute_rse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.sqrt(numerator / denominator) if denominator != 0 else np.nan


# Used for model prediction/evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluation of Models. Metrics are: MSE, MAE, MAPE, SMAPE, RSE, CORR
    """
    predictions = model.predict(X_test).flatten()
    y_test = y_test.flatten()
    print(f"Predictions shape: {predictions.shape}, y_test shape: {y_test.shape}")
    # Ensure no shape mismatches
    if predictions.shape != y_test.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, y_test {y_test.shape}")

    final_results =  {
        "mean_squared_error": mean_squared_error(y_test, predictions),
        "mean_absolute_error":mean_absolute_error(y_test , predictions),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y_test, predictions),
        "smape": smape(y_test, predictions),
        "rse": compute_rse(y_test, predictions),
        "corr": pearsonr(y_test, predictions).statistic
    }

    print(final_results)
    return final_results




def write_loss_history_to_csv(station, model_name, window_size, offset, history, feature_str, label_str):
    """Saves loss history to a unique CSV file including offset and feature set."""
    
     # Ensure the results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
 
    # Define loss history file path inside the results folder
    loss_file = os.path.join(results_dir, f"loss_history_ws{window_size}_offset{offset}_{feature_str}-{label_str}.csv")
    
    # Check if the file already exists
    file_exists = os.path.isfile(loss_file)
    
    # Define CSV headers
    headers = ["Station", "Model", "Features", "Labels", "Offset", "Epoch", "Loss", "Validation Loss"]

    # Open in write mode if file exists (reset each run)
    mode = "a" if file_exists else "w"

    with open(loss_file, mode=mode, newline="") as file:
        writer = csv.writer(file)
        
        # Write headers only if file is new
        if not file_exists:
            writer.writerow(headers)  

        # Write training history
        for epoch, (loss, val_loss) in enumerate(zip(history["loss"], history["val_loss"])):
            writer.writerow([station, model_name, feature_str, label_str, offset, epoch + 1, loss, val_loss])

    print(f"Saved loss history for {model_name} (ws={window_size}, offset={offset}, features={feature_str}, labels={label_str}) on {station} to {loss_file}")

###############################################################################
###############################################################################

    
def write_model_results_to_csv(station, model_name, window_size, offset, performance, feature_str):

    # Ensure the results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"results_ws{window_size}_offset{offset}_{feature_str}.csv")

    file_exists = os.path.isfile(results_file)
    headers = ["Station", "Model", "Features", "Offset", "MSE", "MAE", "MAPE", "SMAPE", "RSE", "CORR"]
    
    # Open in write mode if file exists (reset each run)
    mode = "a" if file_exists else "w"
    
    with open(results_file, mode=mode, newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            station, model_name, feature_str, offset,
            performance.get("MSE"),
            performance.get("MAE"),
            performance.get("MAPE"),
            performance.get("SMAPE"),
            performance.get("RSE"),
            performance.get("CORR")
        ])
    print(f"Saved model results for {model_name} on {station} with {len(feature_str.split('_'))} features to {results_file}")


###############################################################################
###############################################################################
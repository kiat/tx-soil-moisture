
import os
import csv
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
import json
from datetime import datetime

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


def create_run_id(station, model_name, window_size, offset):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_{model_name}_ws{window_size}_off{offset}_{station}"

def write_loss_history_to_csv(station, model_name, window_size, offset, history, feature_str, label_str, run_id=None):
    """Save loss history inside a unique run folder."""

    if run_id is None:
        run_id = create_run_id(station, model_name, window_size, offset)

    run_dir = os.path.join("results", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    loss_file = os.path.join(run_dir, "loss_history.csv")

    headers = ["Station", "Model", "Features", "Labels", "Window Size", "Offset", "Epoch", "Loss", "Validation Loss"]

    with open(loss_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for epoch, (loss, val_loss) in enumerate(zip(history["loss"], history["val_loss"])):
            writer.writerow([
                station, model_name, feature_str, label_str,
                window_size, offset, epoch + 1, loss, val_loss
            ])

    print(f"Saved loss history to {loss_file}")
    return run_id
    
def write_model_results_to_csv(station, model_name, window_size, offset, performance, feature_str, label_str=None, run_id=None, epochs=None, patience=None):
    """Save metrics inside a unique run folder and append summary to master experiment log."""

    if run_id is None:
        run_id = create_run_id(station, model_name, window_size, offset)

    results_dir = "results"
    runs_dir = os.path.join(results_dir, "runs")
    run_dir = os.path.join(runs_dir, run_id)

    os.makedirs(run_dir, exist_ok=True)

    metrics_file = os.path.join(run_dir, "metrics.csv")
    config_file = os.path.join(run_dir, "config.json")
    experiment_log = os.path.join(results_dir, "experiment_log.csv")

    metrics_headers = ["Station", "Model", "Features", "Labels", "Window Size", "Offset", "MSE", "MAE", "MAPE", "SMAPE", "RSE", "CORR"]

    row = [
        station,
        model_name,
        feature_str,
        label_str,
        window_size,
        offset,
        performance.get("MSE"),
        performance.get("MAE"),
        performance.get("MAPE"),
        performance.get("SMAPE"),
        performance.get("RSE"),
        performance.get("CORR")
    ]

    # Save metrics for this specific run
    with open(metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metrics_headers)
        writer.writerow(row)

    # Save config for this run
    config = {
        "run_id": run_id,
        "station": station,
        "model": model_name,
        "window_size": window_size,
        "offset": offset,
        "epochs": epochs,
        "patience": patience,
        "features": feature_str,
        "labels": label_str,
        "run_dir": run_dir
    }

    with open(config_file, mode="w") as file:
        json.dump(config, file, indent=4)

    # Append to master experiment log
    log_exists = os.path.isfile(experiment_log)

    log_headers = ["Run ID", "Date", "Station", "Model", "Features", "Labels", "Window Size", "Offset", "MSE", "MAE", "MAPE", "SMAPE", "RSE", "CORR", "Run Directory"]

    log_row = [
        run_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        station,
        model_name,
        feature_str,
        label_str,
        window_size,
        offset,
        performance.get("MSE"),
        performance.get("MAE"),
        performance.get("MAPE"),
        performance.get("SMAPE"),
        performance.get("RSE"),
        performance.get("CORR"),
        run_dir
    ]

    with open(experiment_log, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not log_exists:
            writer.writerow(log_headers)

        writer.writerow(log_row)

    print(f"Saved metrics to {metrics_file}")
    print(f"Updated master experiment log at {experiment_log}")

    return run_id

def save_experiment_results(
    station,
    model_name,
    window_size,
    offset,
    history,
    performance,
    feature_str,
    label_str,
    epochs=None,
    patience=None
):
    """
    Save all experiment artifacts using one shared run_id:
    - loss_history.csv
    - metrics.csv
    - config.json
    - append row to experiment_log.csv
    """

    run_id = create_run_id(station, model_name, window_size, offset)

    write_loss_history_to_csv(
        station=station,
        model_name=model_name,
        window_size=window_size,
        offset=offset,
        history=history,
        feature_str=feature_str,
        label_str=label_str,
        run_id=run_id
    )

    write_model_results_to_csv(
        station=station,
        model_name=model_name,
        window_size=window_size,
        offset=offset,
        performance=performance,
        feature_str=feature_str,
        label_str=label_str,
        run_id=run_id,
        epochs=epochs,
        patience=patience
    )

    return run_id

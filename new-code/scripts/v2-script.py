# --- Setup: add project root to sys.path so that src is discoverable ---
import sys
from pathlib import Path
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Set project root path
project_root = Path.cwd().parent  # assuming current working directory is 'notebook'
sys.path.insert(0, str(project_root))

# --- Create a timestamped results folder ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = project_root / "results" / f"run_{timestamp}"
results_folder.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved in: {results_folder}")

# --- Standard Imports ---
import matplotlib
matplotlib.use('Agg')  # Prevents from plotting to screen
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# --- Imports from custom modules ---
# from src.config.loaders import load_config, load_station_data
from src.helpers.data_helpers import engineer_data, data_to_X_y, load_config, load_station_data
from src.helpers.scaler import MinMaxScalerWrapper
from src.helpers.plotting import plot_loss, plot_predictions
from src.models.lstm import build_original_simple_lstm_model, build_optimized_lstm_model
from src.models.cnn import build_cnn_model

# --- Load Configuration ---
cfg = load_config()

# --- Define Model Builders ---
model_builders = {
    "Simple LSTM": build_original_simple_lstm_model,
    # "Optimized LSTM": build_optimized_lstm_model,
    # "CNN": build_cnn_model
}

# --- Load and Preprocess Data ---
station_data = load_station_data(cfg)
for station in station_data:
    print(f"Engineering features for {station} ...")
    station_data[station] = engineer_data(station_data[station])

# Select a station for forecasting (example: Station1)
df_forecast = station_data["Station1"]

# Define target column and windowing parameters
TARGET_COLUMN = cfg["model"]["target_column"]  # Example: "SWC_20"
WINDOW_SIZE = cfg["model"]["window_size"]      # Example: 72
OFFSET = cfg["model"].get("offset", 24)        # Default: 24



# Create and fit scaler pipeline
scaler = MinMaxScalerWrapper()
scaled_df = scaler.fit(df_forecast).transform(df_forecast)

# Convert data into sliding windows
X, y = data_to_X_y(scaled_df[TARGET_COLUMN], WINDOW_SIZE, OFFSET)
print("Full sliding windows shapes: X =", X.shape, ", y =", y.shape)

# --- Split into Train, Validation, Test ---
n = len(X)
X_train, y_train = X[:int(n*0.7)], y[:int(n*0.7)]
X_val, y_val = X[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9)]
X_test, y_test = X[int(n*0.9):], y[int(n*0.9):]

print("Split shapes -- X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

# --- Hyperparameters ---
EPOCHS = 3#cfg["lstm"]["epochs"]
BATCH_SIZE = cfg["lstm"]["batch_size"]

# --- Loop over Models ---
results = []  # Store evaluation metrics

for model_name, builder in model_builders.items():
    print(f"\nTraining model: {model_name}")

    # Create unique checkpoint path
    checkpoint_path = results_folder / f"{model_name.replace(' ', '_')}_checkpoint.keras"

    # Define the pipeline: Scaling + LSTM Model
    pipeline = Pipeline([
        ("model", builder(WINDOW_SIZE))  # Build LSTM/CNN model
    ])

    # Define callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path), save_best_only=True, monitor='val_loss', mode='min'
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

    # Train the model
    history = pipeline.named_steps["model"].fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1
    )

    # Plot and save training history
    loss_plot_path = results_folder / f"{model_name.replace(' ', '_')}_loss_plot.png"
    plot_loss(history, save_path=str(loss_plot_path))

    # Reload best model from checkpoint
    best_model = tf.keras.models.load_model(str(checkpoint_path))

    # Make predictions
    predictions_scaled = best_model.predict(X_test).flatten()

    # Reverse scaling
    y_actual_df = scaler.inverse_transform(y_test.reshape(-1, 1), scaled_df, TARGET_COLUMN)
    predictions_rescaled_df = scaler.inverse_transform(predictions_scaled.reshape(-1, 1), scaled_df, TARGET_COLUMN)

    y_actual = y_actual_df[TARGET_COLUMN].values
    predictions_rescaled = predictions_rescaled_df[TARGET_COLUMN].values

    # Compute evaluation metrics
    r2 = r2_score(y_actual, predictions_rescaled)
    mse = mean_squared_error(y_actual, predictions_rescaled)
    mae = mean_absolute_error(y_actual, predictions_rescaled)
    mape = mean_absolute_percentage_error(y_actual, predictions_rescaled)

    results.append({
        "Model": model_name,
        "R2": r2,
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape
    })

    # Save predictions plot
    predictions_plot_path = results_folder / f"{model_name.replace(' ', '_')}_predictions.png"
    plot_predictions(y_actual, predictions_rescaled, TARGET_COLUMN, save_path=str(predictions_plot_path))

    print(f"Finished evaluation for {model_name}:")
    print(f"  R2: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.4f}")

# --- Save Evaluation Metrics ---
results_df = pd.DataFrame(results)
results_df.to_csv(results_folder / "model_comparison_results.csv", index=False)
print("Model performance results saved!")

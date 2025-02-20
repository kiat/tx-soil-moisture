# --- Setup: add project root to sys.path so that src is discoverable ---
import sys
from pathlib import Path
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

project_root = Path.cwd().parent  # assuming current working directory is 'notebook'
sys.path.insert(0, str(project_root))

# --- Create a timestamped results folder ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = project_root / "results" / f"run_{timestamp}"
results_folder.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved in: {results_folder}")
# --- Imports from our modules ---
from src.config.loaders import load_config, load_station_data
from src.data.engineering import engineer_data
from src.data.scaling import scale_df, inverse_scale
from src.data.windowing import data_to_X_y
from src.visualization.plotting import plot_loss, plot_predictions
from src.models.lstm import build_original_simple_lstm_model, build_optimized_lstm_model
from src.models.cnn import build_cnn_model

# --- Standard Setup ---
import matplotlib
matplotlib.use('Agg')  # Prevents from plotting to screen
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.grid'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# --- Load configuration ---
cfg = load_config()

model_builders = {
    "Simple LSTM": build_original_simple_lstm_model,
    "Optimized LSTM": build_optimized_lstm_model,
    "CNN": build_cnn_model
}
# --- Load and preprocess data ---
station_data = load_station_data(cfg)
for station in station_data:
    print(f"Engineering features for {station} ...")
    station_data[station] = engineer_data(station_data[station])

# For forecasting, choose Station1 (for example)
df_forecast = station_data["Station1"]
scaled_df, scaler = scale_df(df_forecast)
# --- Create Sliding Windows ---
TARGET_COLUMN = cfg["model"]["target_column"]  # "SWC_20"
WINDOW_SIZE = cfg["model"]["window_size"]         # 72
OFFSET = cfg["model"].get("offset", 24)             # 24
X, y = data_to_X_y(scaled_df[TARGET_COLUMN], WINDOW_SIZE, OFFSET)
print("Full sliding windows shapes: X =", X.shape, ", y =", y.shape)
# Split into training (70%), validation (20%), test (10%)
n = len(X)
X_train, y_train = X[:int(n*0.7)], y[:int(n*0.7)]
X_val, y_val = X[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9)]
X_test, y_test = X[int(n*0.9):], y[int(n*0.9):]
print("Split shapes -- X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

# --- Hyperparameters ---
EPOCHS = 3# cfg["lstm"]["epochs"]
BATCH_SIZE = cfg["lstm"]["batch_size"]

# --- Loop over Models ---
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

results = []  # To store evaluation metrics for each model

for model_name, builder in model_builders.items():
    print(f"\nTraining model: {model_name}")
    
    # Create a unique checkpoint path for the model within the results folder.
    checkpoint_path = results_folder / f"{model_name.replace(' ', '_')}_checkpoint.keras"
    
    # Build the model using the provided builder and window size.
    model = builder(WINDOW_SIZE)
    
    # Setup callbacks for checkpointing and early stopping.
    cp = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path), save_best_only=True, monitor='val_loss', mode='min'
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    
    # Train the model.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[cp, early_stopping],
        verbose=1
    )
    
    # Optionally, plot and save the training & validation loss.
    loss_plot_path = results_folder / f"{model_name.replace(' ', '_')}_loss_plot.png"
    plot_loss(history, save_path=str(loss_plot_path))
    
    # Reload the best saved model.
    model = tf.keras.models.load_model(str(checkpoint_path))
    
    # Generate predictions on the test set.
    predictions_scaled = model.predict(X_test).flatten()
    preds_inv_df = inverse_scale(predictions_scaled, scaled_df, TARGET_COLUMN)
    predictions_rescaled = preds_inv_df[TARGET_COLUMN].values
    actual_inv_df = inverse_scale(y_test.reshape(-1, 1), scaled_df, TARGET_COLUMN)
    y_actual = actual_inv_df[TARGET_COLUMN].values
    
    # Compute evaluation metrics.
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
    
    # Plot and save predictions vs. actual values.
    predictions_plot_path = results_folder / f"{model_name.replace(' ', '_')}_predictions.png"
    plot_predictions(y_actual, predictions_rescaled, TARGET_COLUMN, save_path=str(predictions_plot_path))
    
    print(f"Finished evaluation for {model_name}:")
    print(f"  R2: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.4f}")
# --- Compile and Save Evaluation Metrics ---
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df)
results_csv_path = results_folder / "model_comparison_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Model performance results saved to {results_csv_path}")

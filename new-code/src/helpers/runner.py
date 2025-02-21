# src/runner.py
import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from helpers.plotting import plot_loss, plot_predictions
import yaml

def run_model(model_name, builder, results_folder, trial_name,
              X_train, y_train, X_val, y_val, X_test, y_test,
              scaled_df, scaler, target_col, epochs, batch_size):
    """
    Trains and evaluates a single model.
    """
    trial_folder = results_folder / trial_name
    trial_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining model: {model_name}")

    # Create checkpoint path
    checkpoint_path = trial_folder / f"{model_name.replace(' ', '_')}_checkpoint.keras"

    # Define callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path), save_best_only=True, monitor='val_loss', mode='min'
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

    # Build and train the model
    model = builder(X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1
    )

    # Save loss plot
    loss_plot_path = trial_folder / f"{model_name.replace(' ', '_')}_loss_plot.png"
    plot_loss(history, save_path=str(loss_plot_path))

    # Load best model from checkpoint
    best_model = tf.keras.models.load_model(str(checkpoint_path))

    # Make predictions
    predictions_scaled = best_model.predict(X_test).flatten()

    # Reverse scaling
    y_actual_df = scaler.inverse_transform(y_test.reshape(-1, 1), scaled_df, target_col)
    predictions_rescaled_df = scaler.inverse_transform(predictions_scaled.reshape(-1, 1), scaled_df, target_col)

    y_actual = y_actual_df[target_col].values
    predictions_rescaled = predictions_rescaled_df[target_col].values

    # Compute evaluation metrics
    r2 = r2_score(y_actual, predictions_rescaled)
    mse = mean_squared_error(y_actual, predictions_rescaled)
    mae = mean_absolute_error(y_actual, predictions_rescaled)
    mape = mean_absolute_percentage_error(y_actual, predictions_rescaled)

    # Save predictions plot
    predictions_plot_path = trial_folder / f"{model_name.replace(' ', '_')}_predictions.png"
    plot_predictions(y_actual, predictions_rescaled, target_col, save_path=str(predictions_plot_path))

    print(f"Finished {model_name}: R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")

    # Save experiment parameters to YAML
    parameters = {
        "Model": model_name,
        "Target Variable": target_col,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Dataset Split": {
            "Train Samples": len(X_train),
            "Validation Samples": len(X_val),
            "Test Samples": len(X_test)
        },
        "Results": {
            "R2": r2,
            "MSE": mse,
            "MAE": mae,
            "MAPE": mape
        }
    }

    parameters_file = trial_folder / "experiment_parameters.yaml"
    with open(parameters_file, "w") as file:
        yaml.dump(parameters, file, default_flow_style=False)

    print(f"Experiment parameters saved in {parameters_file}")


    # Save results to CSV (APPEND MODE)
    results_file = trial_folder / "model_comparison_results.csv"
    results_df = pd.DataFrame([{
        "Model": model_name,
        "R2": r2,
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape
    }])
    results_df.to_csv(results_file, index=False, mode='a', header=not results_file.exists())

    print(f"Results saved in {results_file}")

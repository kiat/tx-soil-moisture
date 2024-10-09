from helper import (
    load_and_preprocess_data, compile_and_fit, compare_models, 
    run_feature_importance_for_model, run_feature_addition_for_model, 
    create_autoregressive_model, create_cnn_model, dense_model, rnn_model, bi_lstm_model, linear_model
)
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Parameters - more generalized
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
WINDOW_SIZE = 24 * 7
SHIFT_AMT = 10
PAT = 3
MAX_EPOCHS = 25
BATCH_SIZE = 128

stations = range(1, 7)  # Stations 1 to 6
target_columns = ["SWC_5", "SWC_10", "SWC_20", "SWC_50"]

# stations = [1]
# target_columns = [ "SWC_5"]

# Loop through each station and each target column
for station in stations:
    for target_col in target_columns:
        print(f"Processing Station {station}, Target Column: {target_col}")

        # Load and preprocess the data
        cur_df, X_train, y_train, X_val, y_val, X_test, y_test, \
        train_dataset, val_dataset, test_dataset, \
        train_steps, val_steps, test_steps = load_and_preprocess_data(
            station, target_col, TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT, BATCH_SIZE
        )
        # Define models
        models = {
            "biLSTM": bi_lstm_model,
            "linear": linear_model,
            "autoregressive": create_autoregressive_model(X_train.shape[-2:]),
            "dense": dense_model,
            "RNN": rnn_model,
            "CNN": create_cnn_model(X_train.shape[-2:])
        }

        # # ARIMA Model
        # arima_order = (1, 0, 0)  # p, d, q
        # train_size = int(len(y_train) * 0.8)
        # train_data_arima = y_train[:train_size]
        # test_data_arima = y_test
        # arima_mse, arima_mae, arima_mape = run_arima_model(train_data_arima, test_data_arima, order=arima_order)

        # Train each model
        model_histories = {}
        for model_name, model in models.items():
            print(f"Training {model_name} for Station {station}, Target: {target_col}...")
            history = compile_and_fit(
                model, train_dataset, train_steps, val_dataset, val_steps, 
                batch_size=BATCH_SIZE, model_name=f"{model_name}_Station{station}_{target_col}",
                patience=PAT, max_epochs=MAX_EPOCHS
            )
            model_histories[model_name] = history

        # Collect model results
        model_results = {}
        for model_name, history in model_histories.items():
            model_results[model_name] = {
                "MSE": history.history['val_mean_squared_error'][-1],
                "MAE": history.history['val_mean_absolute_error'][-1],
                "MAPE": history.history['val_mean_absolute_percentage_error'][-1]
            }

        # Compare models
        print(f"Comparing models for Station {station}, Target: {target_col}")
        compare_models(model_results)


        # Convert the dictionary to a DataFrame with the desired column order
        output_dir = 'results'
        results_df = pd.DataFrame.from_dict(model_results, orient='index')
        results_df = results_df[["MAE", "MAPE", "MSE", "R2"]]  # Ensure the correct column order

        # Define the output file path
        output_file = os.path.join(output_dir, f"results_Station{station}_{target_col}.csv")

        # Save the DataFrame to a CSV file with a header
        results_df.to_csv(output_file, index_label='Model')

        # Feature names (excluding the target column)
        feature_names = [col for col in cur_df.columns if col != target_col]

        # Run feature importance analysis for RNN model
        run_feature_importance_for_model(
            cur_df, feature_names, target_col, rnn_model, "RNN",
            TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT, BATCH_SIZE, PAT, MAX_EPOCHS
        )

        # Run feature importance analysis for Dense model
        run_feature_importance_for_model(
            cur_df, feature_names, target_col, dense_model, "Dense",
            TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT, BATCH_SIZE, PAT, MAX_EPOCHS
        )

        # Run feature addition analysis for RNN model
        run_feature_addition_for_model(
            cur_df, feature_names, target_col, rnn_model, "RNN",
            TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT, BATCH_SIZE, PAT, MAX_EPOCHS
        )

        # Run feature addition analysis for Dense model
        run_feature_addition_for_model(
            cur_df, feature_names, target_col, dense_model, "Dense",
            TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT, BATCH_SIZE, PAT, MAX_EPOCHS
        )

        print(f"Completed processing for Station {station}, Target Column: {target_col}\n")
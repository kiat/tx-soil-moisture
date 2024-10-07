from helper import (
    load_data, preprocess_data, generate_batches, compile_and_fit, plot_single_pred, 
    create_autoregressive_model, run_arima_model, compare_models, dense_model, 
    rnn_model, create_cnn_model, bi_lstm_model, linear_model
)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# parameters - more generalized
STATION = 1
TARGET_COL = "SWC_5"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
WINDOW_SIZE = 24 * 7
SHIFT_AMT = 10
PAT = 3
MAX_EPOCHS = 25
BATCH_SIZE = 128

# load and preprocess data

station_filepath = f'../datasets/Simulate_Cleaned_Merged/Station{STATION}_simulated_cleaned_merged_data.csv'
dfs = load_data(station_filepath)
cur_df = dfs["cur_station"]
cur_df = cur_df[[col for col in cur_df.columns if not col.startswith('SWC') or col == TARGET_COL]]


X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(cur_df, TARGET_COL, TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT)

train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=BATCH_SIZE)
val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=BATCH_SIZE)
test_dataset, test_steps = generate_batches(X_test, y_test, batch_size=BATCH_SIZE)

# train and plot models
# history_bilstm = compile_and_fit(bi_lstm_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="biLSTM", patience=PAT, max_epochs=MAX_EPOCHS)

# Linear
# history_linear = compile_and_fit(linear_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="linear", patience=PAT, max_epochs=MAX_EPOCHS)

# Autoregressive
# autoregressive_model = create_autoregressive_model(X_train.shape[-2:])
# history_ar = compile_and_fit(autoregressive_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="autoregressive", patience=PAT, max_epochs=MAX_EPOCHS)

# Dense
# history_dense = compile_and_fit(dense_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="dense", patience=PAT, max_epochs=MAX_EPOCHS)

# RNN
# history_rnn = compile_and_fit(rnn_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="RNN", patience=PAT, max_epochs=MAX_EPOCHS)

# CNN
# cnn_model = create_cnn_model(X_train.shape[-2:])

# ARIMA Model
# arima_order = (1, 0, 0)  # p, d, q
# train_size = int(len(y_train) * 0.8)
# train_data_arima = y_train[:train_size]
# test_data_arima = y_test
# arima_mse, arima_mae, arima_mape = run_arima_model(train_data_arima, test_data_arima, order=arima_order)



# # compare models, this is without removing anything
# model_results = {
#     "BiLSTM": history_bilstm.history['val_mean_squared_error'][-1],  # Add last epoch's MSE from history
#     "Linear": history_linear.history['val_mean_squared_error'][-1],
#     "Autoregressive": history_ar.history['val_mean_squared_error'][-1],
#     "dense": history_dense.history['val_mean_squared_error'][-1],
#     "ARIMA": arima_mse,
#     "RNN": history_rnn.history['val_mean_squared_error'][-1],
#     "CNN": history_cnn.history['val_mean_squared_error'][-1]
# }
# compare_models(model_results)



# Feature Importance Analysis
# Removing features to find the most important feature in any given model
feature_names = [col for col in cur_df.columns if col != TARGET_COL]

# Retrain the specified model after removing each feature and track MSE, MAE, MAPE
def feature_importance_analysis(features, cur_df, target_col, model, model_name):
    mse_results = {}
    mae_results = {}
    mape_results = {}

    for feature in features:
        print(f"Training {model_name} without feature: {feature}")
        cur_df_modified = cur_df.drop(columns=[feature])
        
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
            cur_df_modified, target_col, TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT
        )
        train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=BATCH_SIZE)
        val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=BATCH_SIZE)
        test_dataset, test_steps = generate_batches(X_test, y_test, batch_size=BATCH_SIZE)

        # Re-train the specified model without the feature
        history = compile_and_fit(
            model, train_dataset, train_steps, val_dataset, val_steps,
            batch_size=BATCH_SIZE, model_name=f"{model_name}_without_{feature}",
            patience=PAT, max_epochs=MAX_EPOCHS
        )

        # Extract the last values of MSE, MAE, and MAPE from the training history
        mse = history.history['val_mean_squared_error'][-1]
        mae = history.history['val_mean_absolute_error'][-1]
        mape = history.history['val_mean_absolute_percentage_error'][-1]

        # Store results
        mse_results[feature] = mse
        mae_results[feature] = mae
        mape_results[feature] = mape
    
    return mse_results, mae_results, mape_results

# Perform feature importance analysis for any model and sort results by MSE, MAE, and MAPE
def run_feature_importance_for_model(model, model_name):
    mse_results_without_features, mae_results_without_features, mape_results_without_features = feature_importance_analysis(
        feature_names, cur_df, TARGET_COL, model, model_name
    )

    # Display and sort by MSE
    print(f"\nMSE results after removing each feature for {model_name}: (higher indicates more important)") 
    sorted_mse = sorted(mse_results_without_features.items(), key=lambda x: x[1], reverse=True)
    for feature, mse in sorted_mse:
        print(f"Feature: {feature}, MSE: {mse}")

    # Display and sort by MAE
    print(f"\nMAE results after removing each feature for {model_name}:")
    sorted_mae = sorted(mae_results_without_features.items(), key=lambda x: x[1], reverse=True)
    for feature, mae in sorted_mae:
        print(f"Feature: {feature}, MAE: {mae}")

    # Display and sort by MAPE
    print(f"\nMAPE results after removing each feature for {model_name}:")
    sorted_mape = sorted(mape_results_without_features.items(), key=lambda x: x[1], reverse=True)
    for feature, mape in sorted_mape:
        print(f"Feature: {feature}, MAPE: {mape}")

# Run feature importance analysis for RNN model
run_feature_importance_for_model(rnn_model, "RNN")

# Run feature importance analysis for Dense model
# run_feature_importance_for_model(dense_model, "Dense")



from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Feature importance analysis by adding one feature at a time to see its effect on MSE, MAE, and MAPE
def feature_addition_analysis(features, cur_df, target_col, model, model_name):
    mse_results = {}
    mae_results = {}
    mape_results = {}

    for feature in features:
        print(f"Training {model_name} with only feature: {feature}")
        # Use only the selected feature along with the target column
        cur_df_modified = cur_df[[feature, target_col]]
        
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
            cur_df_modified, target_col, TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT
        )
        train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=BATCH_SIZE)
        val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=BATCH_SIZE)
        test_dataset, test_steps = generate_batches(X_test, y_test, batch_size=BATCH_SIZE)

        # Re-train the specified model with the single feature
        history = compile_and_fit(
            model, train_dataset, train_steps, val_dataset, val_steps,
            batch_size=BATCH_SIZE, model_name=f"{model_name}_with_only_{feature}",
            patience=PAT, max_epochs=MAX_EPOCHS
        )

        # Extract the last values of MSE, MAE, and MAPE from the training history
        mse = history.history['val_mean_squared_error'][-1]
        mae = history.history['val_mean_absolute_error'][-1]
        mape = history.history['val_mean_absolute_percentage_error'][-1]

        # Store results
        mse_results[feature] = mse
        mae_results[feature] = mae
        mape_results[feature] = mape
    
    return mse_results, mae_results, mape_results

# Perform feature addition analysis for any model and sort results by MSE, MAE, and MAPE
def run_feature_addition_for_model(model, model_name):
    mse_results_with_features, mae_results_with_features, mape_results_with_features = feature_addition_analysis(
        feature_names, cur_df, TARGET_COL, model, model_name
    )

    # Display and sort by MSE
    print(f"\nMSE results after adding each feature for {model_name}: (lower indicates more important)")
    sorted_mse = sorted(mse_results_with_features.items(), key=lambda x: x[1], reverse=False)
    for feature, mse in sorted_mse:
        print(f"Feature: {feature}, MSE: {mse}")

    # Display and sort by MAE
    print(f"\nMAE results after adding each feature for {model_name}:")
    sorted_mae = sorted(mae_results_with_features.items(), key=lambda x: x[1], reverse=False)
    for feature, mae in sorted_mae:
        print(f"Feature: {feature}, MAE: {mae}")

    # Display and sort by MAPE
    print(f"\nMAPE results after adding each feature for {model_name}:")
    sorted_mape = sorted(mape_results_with_features.items(), key=lambda x: x[1], reverse=False)
    for feature, mape in sorted_mape:
        print(f"Feature: {feature}, MAPE: {mape}")


# run feature addition analysis for dense model
# run_feature_addition_for_model(dense_model, "Dense")

# run feature addition analysis for rnn model
# run_feature_addition_for_model(rnn_model, "RNN")





# TODO: make model look slightly cleaner, add MSE, MAE, and MAPE
# TODO: Format into how it looks in /code/ folder
# TODO: drop other SWC
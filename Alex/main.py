from helper import (
    load_data, preprocess_data, generate_batches, compile_and_fit, plot_single_pred, 
    create_autoregressive_model, run_arima_model, compare_models, dense_model, 
    rnn_model, create_cnn_model, bi_lstm_model
)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# parameters - more generalized
TARGET_COL = "SWC_5"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
WINDOW_SIZE = 24 * 7
SHIFT_AMT = 10
PAT = 3
MAX_EPOCHS = 25
BATCH_SIZE = 128

# load and preprocess data

dfs = load_data('../datasets/Simulate_Cleaned_Merged/Station1_simulated_cleaned_merged_data.csv')
cur_df = dfs["Station1"]

X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(cur_df, TARGET_COL, TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT)

train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=BATCH_SIZE)
val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=BATCH_SIZE)
test_dataset, test_steps = generate_batches(X_test, y_test, batch_size=BATCH_SIZE)

# train and plot models
# bilstm
history_bilstm = compile_and_fit(bi_lstm_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="biLSTM", patience=PAT, max_epochs=MAX_EPOCHS)
plot_single_pred(bi_ltm_model, 'BiLSTM', test_dataset, test_steps, y_test, batch_size=BATCH_SIZE)

#linear
history_linear = compile_and_fit(linear_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="linear", patience=PAT, max_epochs=MAX_EPOCHS)

#autoregressive
autoregressive_model = create_autoregressive_model(X_train.shape[-2:])
history_ar = compile_and_fit(autoregressive_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="autoregressive", patience=PAT, max_epochs=MAX_EPOCHS)

#Dense
history_dense = compile_and_fit(dense_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="dense", patience=PAT, max_epochs=MAX_EPOCHS)

# RNN
history_rnn = compile_and_fit(rnn_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="RNN", patience=PAT, max_epochs=MAX_EPOCHS)

# CNN
cnn_model = create_cnn_model(X_train.shape[-2:])
history_cnn = compile_and_fit(cnn_model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name="CNN", patience=PAT, max_epochs=MAX_EPOCHS)


# ARIMA Model
arima_order = (1, 0, 0) #p,d,q
train_size = int(len(y_train) * 0.8) 
train_data_arima = y_train[:train_size]  
test_data_arima = y_test 
arima_mse = run_arima_model(train_data_arima, test_data_arima, order=arima_order)

# compare models, this is without removing anything
model_results = {
    "BiLSTM": history_bilstm.history['val_mean_squared_error'][-1],  # Add last epoch's MSE from history
    "Linear": history_linear.history['val_mean_squared_error'][-1],
    "Autoregressive": history_ar.history['val_mean_squared_error'][-1],
    "dense": history_dense.history['val_mean_squared_error'][-1],
    "ARIMA": arima_mse,
    "RNN": history_rnn.history['val_mean_squared_error'][-1],
    "CNN": history_cnn.history['val_mean_squared_error'][-1]
}
compare_models(model_results)




# Feature Importance Analysis
# removing features to find the most important feature in any given model
feature_names = [col for col in cur_df.columns if col != TARGET_COL]

# retrain the specified model after removing each feature
def feature_importance_analysis(features, cur_df, target_col, model, model_name):
    mse_results = {}

    for feature in features:
        print(f"training {model_name} without feature: {feature}")
        cur_df_modified = cur_df.drop(columns=[feature])
        
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(cur_df_modified, target_col, TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT)
        train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=BATCH_SIZE)
        val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=BATCH_SIZE)
        test_dataset, test_steps = generate_batches(X_test, y_test, batch_size=BATCH_SIZE)

        # re-train the specified model without the feature
        history = compile_and_fit(model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name=f"{model_name}_without_{feature}", patience=PAT, max_epochs=MAX_EPOCHS)

        # calculate mse on the test set
        mse = history.history['val_mean_squared_error'][-1]
        mse_results[feature] = mse
    
    return mse_results

# perform feature importance analysis for any model
def run_feature_importance_for_model(model, model_name):
    mse_results_without_features = feature_importance_analysis(feature_names, cur_df, TARGET_COL, model, model_name)

    # display results
    print(f"\nmse results after removing each feature for {model_name}:")
    for feature, mse in mse_results_without_features.items():
        print(f"feature: {feature}, mse: {mse}")

    # sort by mse to identify which feature caused the biggest change
    sorted_mse = sorted(mse_results_without_features.items(), key=lambda x: x[1], reverse=True)
    print(f"\nsorted mse (higher mse indicates more important feature in {model_name}):")
    for feature, mse in sorted_mse:
        print(f"feature: {feature}, mse: {mse}")

# run feature importance analysis for rnn model
run_feature_importance_for_model(rnn_model, "RNN")

# # run feature importance analysis for dense model
run_feature_importance_for_model(dense_model, "Dense")




# adding one fature at a time to see its effect on mse
# feature importance analysis by adding one feature at a time to see its effect on mse
def feature_addition_analysis(features, cur_df, target_col, model, model_name):
    mse_results = {}

    for feature in features:
        print(f"training {model_name} with only feature: {feature}")
        # Use only the selected feature along with the target column
        cur_df_modified = cur_df[[feature, target_col]]
        
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(cur_df_modified, target_col, TRAIN_SPLIT, VAL_SPLIT, WINDOW_SIZE, SHIFT_AMT)
        train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=BATCH_SIZE)
        val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=BATCH_SIZE)
        test_dataset, test_steps = generate_batches(X_test, y_test, batch_size=BATCH_SIZE)

        # re-train the specified model with the single feature
        history = compile_and_fit(model, train_dataset, train_steps, val_dataset, val_steps, batch_size=BATCH_SIZE, model_name=f"{model_name}_with_only_{feature}", patience=PAT, max_epochs=MAX_EPOCHS)

        # calculate mse on the test set
        mse = history.history['val_mean_squared_error'][-1]
        mse_results[feature] = mse
    
    return mse_results

# perform feature addition analysis for any model
def run_feature_addition_for_model(model, model_name):
    mse_results_with_features = feature_addition_analysis(feature_names, cur_df, TARGET_COL, model, model_name)

    # display results
    print(f"\nmse results after adding each feature for {model_name}:")
    for feature, mse in mse_results_with_features.items():
        print(f"feature: {feature}, mse: {mse}")

    # sort by mse to identify which feature resulted in the best performance
    sorted_mse = sorted(mse_results_with_features.items(), key=lambda x: x[1], reverse=False)
    print(f"\nsorted mse (lower mse indicates more important feature in {model_name}):")
    for feature, mse in sorted_mse:
        print(f"feature: {feature}, mse: {mse}")

# run feature addition analysis for dense model
run_feature_addition_for_model(dense_model, "Dense")

# run feature addition analysis for rnn model
run_feature_addition_for_model(rnn_model, "RNN")





#TODO: add arima model >
#TODO: analyze all models by seeing which performs best (MSE, loss, recall, etc) >
#TODO: see which feature is most important (add features one at a time to see)
#TODO: get .py file to work>
#TODO: add paragraph report to show what I've learned what i discovered

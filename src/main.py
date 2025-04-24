import argparse
import os
import csv
import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError


from scipy.stats import pearsonr

import models as model_module

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt

import models as model_module
from core.model_entry import ModelEntry
from core.data_helpers import read_and_process_csvs, engineer_features, split_and_stack_data, normalize_features, data_to_X_y, concatenate_with_gaps, plot_split_timeline
from core.evaluation_helpers import evaluate_model, write_loss_history_to_csv, write_model_results_to_csv




    
def main(args):
    

    # engineer_and_save_data()
    stations = ['Station1', 'Station2', 'Station3', 'Station4', 'Station5', 'Station6']
    target_station = stations[-1]  # By default, use the last station as the target

    # Load and process raw CSVs in-memory.
    # The data is in `Revised_Final_Data/` folder.
    # The CSVs are named like: Station1_Revised_Final_Data.csv
    # The data is in the format: Date, SWC_5, SWC_10, SWC_20, SWC_50, T_5, T_10, T_20, T_50, Ppt
    # The methods in `data_helpers.py` will read the CSVs, clean the data, and engineer features.
    raw_dfs = read_and_process_csvs()
    engineered_dfs = engineer_features(raw_dfs)
    
    
    # Split data:
    # - `val_df` → Past years of target station (validation)
    # - `test_df` → Current year of target station (testing)
    # - `train_dfs` → All other stations (training)
    engineered_dfs, val_df, test_df = split_and_stack_data(engineered_dfs, test_station_name=target_station, remove_met=False)

    # OPTIONAL: Print features in the data
    if 0:
        print("FEATURES IN THE DATA\n")
        for station, df in engineered_dfs.items():
            print(f"--- {station} ---")
            print(df.describe())  # Summary statistics
            
    # Use the features specified in the CLI or default to the ones below
    # The features are: SWC_5, SWC_10, SWC_20, SWC_50, T_5, T_10, T_20, T_50, Ppt
    all_features = args.features.split(',') if args.features else ['SWC_20', 'T_20', 'Ppt', 'Tair', 'Wx', 'Wy']

    # OPTIONAL: Instead of running the models, visualize the split if and only if the --visualize flag is set
    if args.visualize:
        # Prepare unscaled Date-indexed DataFrames
        val_df_plot = val_df.set_index("Date")
        test_df_plot = test_df.set_index("Date")
        train_df_plot = concatenate_with_gaps([
            df.set_index("Date") for name, df in engineered_dfs.items()
            if name != target_station
        ])


        # Choose a feature to visualize
        feature_for_plot = all_features[0]
        plot_split_timeline(train_df_plot, val_df_plot, test_df_plot, feature=feature_for_plot)

        return  # Exit before training


    # Prepare validation & test sets
    scaled_val, _ = normalize_features(val_df, all_features)
    X_val, y_val = data_to_X_y(scaled_val, args.window_size, args.offset)

    scaled_test, _ = normalize_features(test_df, all_features)
    X_test, y_test = data_to_X_y(scaled_test, args.window_size, args.offset)

    # Prepare training data: merge all stations except target station
    train_data = []  
    for train_station in [s for s in stations if s != target_station]:
        print(f"Adding {train_station} to training pool...")
        scaled_train, _ = normalize_features(engineered_dfs[train_station], all_features)
        X_train, y_train = data_to_X_y(scaled_train, args.window_size, args.offset)
        train_data.append((X_train, y_train))

    # Merge all training data into a single dataset
    X_train = np.concatenate([data[0] for data in train_data], axis=0)
    y_train = np.concatenate([data[1] for data in train_data], axis=0)

    print(f"\nTraining on {len(stations)-1} stations and testing on {target_station}...\n")

    # Ensure y is the right shape
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Print for debugging
    if 0:
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"Features being used: {all_features}")

    # Specify & create saved models directory
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)


    ####### DYNAMICALLY BUILD & RETRIEVE MODELS #######
    # We took an object oriented approach to defining and queuing models to be trained.
    # Model architectures are defined by `compile functions` in `models.py`
    # The compile functions are named in the form `compile_*`, like: `compile_lstm`, `compile_attention_lstm`, etc.
    # The `compile_*` functions return a compiled model.

    # `ModelEntry` objects, as defined in `core/model_entry.py`, are created for each model architecture.
    # Each `ModelEntry` object contains its internal name, display name, and compile function.

    # To create additional models, simply add a new compile function in `models.py` and it will be automatically included.


    # This section takes takes the CLI input for model names and creates `ModelEntry` objects for each
    # The models will then be executed via `model.fit()` in the loop below.
    # The user can specify models like: "LSTM, CNN, AttentionLSTM"

    ### NORMALIZATION
    # We recognize that the user may not know the exact names of the models in `models.py`.
    # For example, they may enter either "Attention_LSTM" or "AttentionLSTM" in order to refer to the same model.
    # The code below handles this by `normalizing` the model names.
    # It uses the `normalize_id` function to convert both the user input and the model names in `models.py` to a common format.
    # This way, the user can enter model names in any format and they will be matched correctly.
    # The function removes underscores and converts to lowercase.
    # For example, "attentionLSTM", "ATTENTION_LSTM", "AttentionLstm", will all be normalized to "attentionlstm".
    def normalize_id(name: str) -> str:
        return name.lower().replace("_", "")



    # Normalize the model names from the command line
    requested_ids = set(normalize_id(n) for n in args.model_names.split(","))

    # Now we build a queue of models to be processed
    process_queue = {}

    # To do this, we iterate over the `dir(model_module)` to find all compile functions
    # If any of them share a normalized name with the user input, we add them to the queue
    for name in dir(model_module):
        if name.startswith("compile_"):
            # Normalize the model name from the compile function
            internal_name = name.replace("compile_", "") 
            model_id = normalize_id(internal_name)

            # Check if the model_id is in the requested_ids
            # If so, create a ModelEntry object and add it to the process_queue
            if model_id in requested_ids:
                compile_fn = getattr(model_module, name)
                process_queue[model_id] = ModelEntry(internal_name, compile_fn)
    print(f"Models to be processed: {process_queue.keys()}")



    # Now we can iterate over the process_queue and train each model
    for model_id, model_entry in process_queue.items():
        # Create the model
        # The model is built using the `build` method of the ModelEntry object
        # The input shape is (window_size, number of features)
        model = model_entry.build((args.window_size, len(all_features)))

        print(f"\nTraining {model_entry} across stations...\n")  # __str__ used automatically

        # This is an edge case, but we need to handle the Baseline model separately
        if model_id == "baseline":
            model.fit(X_train, y_train)
            performance = evaluate_model(model, X_test, y_test)

            note_path = os.path.join(model_dir, f"model_{model_id}_NOTE.txt")
            with open(note_path, "w") as f:
                f.write("Baseline model - no weights saved.\n")

            feature_str = '_'.join(all_features)
            write_model_results_to_csv(target_station, str(model_entry), args.window_size, args.offset, performance, feature_str)
            print(f"{model_entry} Final Test Loss: {performance['mean_squared_error']}\n")
            continue

        # Otherwise, compile and train model
        model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=0.001),
            metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)]
        )

        print("\nFinal Evaluation on Test Set...\n")
        performance = evaluate_model(model, X_test, y_test)

        # Save trained model
        feature_str = '_'.join(all_features)
        main_name = f"model_{model_id}_ws{args.window_size}_offset{args.offset}_{feature_str}"
        model_path = os.path.join(model_dir, f"{main_name}.keras")
        model.save(model_path)
        print(f"{model_entry} saved at {model_path}")

        # Save results
        write_model_results_to_csv(target_station, str(model_entry), args.window_size, args.offset, performance, feature_str)
        write_loss_history_to_csv(target_station, str(model_entry), args.window_size, args.offset, history.history, feature_str)

        print(f"{model_entry} Final Test Loss: {performance['mean_squared_error']}\n")

    print("All Runs Complete! All results saved.")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train various models for time series prediction.")
    parser.add_argument('--window_size', type=int, default=168, help='Window size for input data')
    parser.add_argument('--offset', type=int, default=24, help='Offset for prediction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument("--features", type=str, default="SWC_20,T_20,Ppt,Tair,Wx,Wy", help="Comma-separated list of features to use in training")

    # NOTE: See the `models.py` file for the full list of models
    # The normalization section above will ensure that the user can enter model names in any format
    # For example, they can enter either "Attention_LSTM" or "AttentionLSTM" in order to refer to the same model.
    # The code will automatically normalize the model names and match them to the compile functions in `models.py`
    parser.add_argument("--model_names", type=str, default="LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline",  help="Comma-separated list of Models short form like LSTM,CNN")

    # If this is set, the run is repurposed.
    # No models will be trained, but the train/val/test splits will be visualized and saved to the results folder.
    parser.add_argument('--visualize', action='store_true', help='If true, plots train/val/test splits instead of running models')

    args = parser.parse_args()
    main(args)

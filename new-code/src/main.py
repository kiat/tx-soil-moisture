# src/main.py
import os
import pandas as pd
import datetime
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')  # Prevents from plotting to screen

# Suppress TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from helpers.data_splitter import split_by_year
from helpers.data_helpers import load_config, load_and_engineer_data, data_to_X_y
from helpers.scaler import MinMaxScalerWrapper
from runner import run_model
from models.lstm import build_original_simple_lstm_model

# Load configuration
cfg = load_config()
station_data = load_and_engineer_data(cfg)

# Parse command-line arguments for overriding config values
parser = argparse.ArgumentParser(description="Run soil moisture prediction experiments.")
parser.add_argument("--window_sizes", nargs="+", type=int, help="List of window sizes to test.")
parser.add_argument("--offset", type=int, help="Forecasting offset.")
parser.add_argument("--swc_list", nargs="+", help="List of SWC columns.")
parser.add_argument("--station_list", nargs="+", help="List of stations.")
parser.add_argument("--epochs", type=int, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, help="Batch size for training.")
args = parser.parse_args()

# Allow overriding config values from command-line arguments
WINDOW_SIZES = args.window_sizes if args.window_sizes else cfg["experiment"]["window_sizes"]
OFFSET = args.offset if args.offset else cfg["experiment"]["offset"]
SWC_LIST = args.swc_list if args.swc_list else cfg["experiment"]["swc_list"]
STATION_LIST = args.station_list if args.station_list else cfg["experiment"]["station_list"]
EPOCHS = args.epochs if args.epochs else cfg["hyperparameters"]["epochs"]
BATCH_SIZE = args.batch_size if args.batch_size else cfg["hyperparameters"]["batch_size"]

# Print overridden values
print(f"Using WINDOW_SIZES: {WINDOW_SIZES}")
print(f"Using OFFSET: {OFFSET}")
print(f"Using SWC_LIST: {SWC_LIST}")
print(f"Using STATION_LIST: {STATION_LIST}")
print(f"Using EPOCHS: {EPOCHS}")
print(f"Using BATCH_SIZE: {BATCH_SIZE}")

# 🔹 Create results folder **ONE LEVEL ABOVE `src/`**
project_root = Path(__file__).resolve().parent.parent  # Moves up one level from `src/`
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = project_root / "results" / f"run_{timestamp}"
results_folder.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved in: {results_folder}")

# Loop over WINDOW_SIZES first (outermost loop)
trial_num = 1
for WINDOW_SIZE in WINDOW_SIZES:
    print(f"\n🔹 Using WINDOW_SIZE = {WINDOW_SIZE}")

    for swc in SWC_LIST:
        for station in STATION_LIST:
            # Load and preprocess data
            df_forecast = station_data[station]
            scaler = MinMaxScalerWrapper().fit(df_forecast)
            scaled_df = scaler.transform(df_forecast)

            # Convert data into supervised learning format
            X, y = data_to_X_y(scaled_df[swc], WINDOW_SIZE, OFFSET)

            # Split data by year
            X_train, y_train, X_val, y_val, X_test, y_test = split_by_year(
                df=scaled_df, 
                target_col=swc, 
                train_years=[2015, 2016, 2017, 2018], 
                val_years=[2019], 
                test_years=[2020]
            )

            print(f"Data split for {station} - {swc} - WINDOW_SIZE {WINDOW_SIZE}:")
            print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

            for model_name, builder in {"Simple LSTM": build_original_simple_lstm_model}.items():
                trial = f"Trial{trial_num}"
                print(f"Running {model_name} - Station: {station}, SWC: {swc}, WINDOW_SIZE: {WINDOW_SIZE}")

                run_model(
                    model_name, builder, results_folder, trial,
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    scaled_df, scaler, swc, EPOCHS, BATCH_SIZE
                )

                trial_num += 1

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


# Import custom modules
from helpers.arg_parser import get_config_parameters
from helpers.data_splitter import split_by_year, leave_one_station_one_year_out
from helpers.data_helpers import  load_and_engineer_data
from helpers.runner import run_model
from helpers.scaler import MinMaxScalerWrapper

# Load configuration
params, cfg = get_config_parameters()       # This function is defined in arg_parser.py
station_data = load_and_engineer_data(cfg)  # This function is defined in data_helpers.py

# Assign global-like variables
SPLIT_STRATEGY = params["SPLIT_STRATEGY"]
WINDOW_SIZES = params["WINDOW_SIZES"]
OFFSET = params["OFFSET"]
SWC_LIST = params["SWC_LIST"]
STATION_LIST = params["STATION_LIST"]
TEST_YEAR = params["TEST_YEAR"]
EPOCHS = params["EPOCHS"]
BATCH_SIZE = params["BATCH_SIZE"]

# Print overridden values
print(f" 🔸 Using SPLIT_STRATEGY: {SPLIT_STRATEGY}")
print(f" 🔸 Using WINDOW_SIZES: {WINDOW_SIZES}")
print(f" 🔸 Using OFFSET: {OFFSET}")
print(f" 🔸 Using SWC_LIST: {SWC_LIST}")
print(f" 🔸 Using STATION_LIST: {STATION_LIST}")
print(f" 🔸 Using TEST_YEAR: {TEST_YEAR}")
print(f" 🔸 Using EPOCHS: {EPOCHS}")
print(f" 🔸 Using BATCH_SIZE: {BATCH_SIZE}")

# Create timestamped results folder
project_root = Path(__file__).resolve().parent.parent  # Moves up one level from `src/`
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = project_root / "results" / f"run_{timestamp}"
results_folder.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved in: {results_folder}")

# TODO: Standard process of creating & introducing models
from models.lstm import build_original_simple_lstm_model
model_dict = {
    "Simple LSTM": build_original_simple_lstm_model
}

'''
For each SWC, we loop through each WINDOW_SIZE/MODEL pair.
For each pair, we loop through each test station.
For each test station, we split the data and run the model.

Here, you can either split up the data by year,
or you can leave one station out for a specific year.
'''
trial_num = 1
for swc in SWC_LIST:
    print(f"\n🔹 Running Experiments for SWC: {swc}")

    for WINDOW_SIZE in WINDOW_SIZES:
        print(f"\n🔹 Using WINDOW_SIZE = {WINDOW_SIZE}")

        for model_name, builder in model_dict.items():
            print(f"\n🔹 Training Model: {model_name}")

            for test_station in STATION_LIST:
                
                # Initialize and fit MinMaxScalerWrapper
                scaler = MinMaxScalerWrapper().fit(station_data[test_station])
                scaled_df = scaler.transform(station_data[test_station])

                if SPLIT_STRATEGY == "split_by_year":
                    X_train, y_train, X_val, y_val, X_test, y_test = split_by_year(
                        df=station_data[test_station], 
                        target_col=swc, 
                        train_years=[2015, 2016, 2017, 2018], 
                        val_years=[2019], 
                        test_years=[2020]
                    )
                else:
                    print(f"🔹 Leaving Out {test_station} for Year {TEST_YEAR} (as test set)")
                    X_train, y_train, X_test, y_test = leave_one_station_one_year_out(
                        df_dict=station_data,
                        target_col=swc,
                        test_station=test_station,
                        test_year=TEST_YEAR,
                        window_size=WINDOW_SIZE,
                        offset=OFFSET
                    )
                    X_val, y_val = None, None  # No validation set in leave-one-out

                print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

                trial = f"Trial{trial_num}"
                print(f"Running {model_name} - Test Station: {test_station}, Test Year: {TEST_YEAR}, SWC: {swc}, WINDOW_SIZE: {WINDOW_SIZE}")

                run_model(
                    model_name, builder, results_folder, trial,
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    scaled_df, scaler, swc, EPOCHS, BATCH_SIZE
                )

                trial_num += 1
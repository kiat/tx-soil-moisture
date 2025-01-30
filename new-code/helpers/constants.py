# File: helpers/constants.py

# Paths
DATA_FOLDER = "../data/processed/Revised_Final_Data"

# Column Names
TARGET_COLUMN = "SWC_10"
TIME_COLUMNS = ["hour", "day", "month", "year"]
CYCLIC_COLUMNS = ["day_sin", "day_cos", "month_sin", "month_cos"]

# ARIMA Parameters
ARIMA_ORDER = (5, 1, 2)

# LSTM Parameters
WINDOW_SIZE = 48
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 64

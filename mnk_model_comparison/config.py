# config.py

from sklearn.model_selection import ParameterGrid

INPUT_PATH = 'C:/Users/mnabh/Research - Soil Moisture Content/tx-soil-moisture/datasets/Revised_Final_Data/Station3_Revised_Final_Data.csv'

# === Models to Run ===
MODELS_TO_RUN = ['cnn', 'lstm', 'arima', 'arimax']
LOG_TO_CSV = True

# === Data Windowing Parameters ===
INPUT_WINDOW = 72                     # 3 days (hourly data)
OUTPUT_HORIZON = 48                   # Default: 2 days ahead
OUTPUT_HORIZONS = [48, 72, 168]       # Test 2-day, 3-day, 1-week horizons
TEST_FULL = True                     # Whether to test all output horizons or just one

HIGH_CORR_FILTER = False
THRESHOLD = 0.95

# === Date Ranges ===
SEASONS = {
    "spring": ("03-01", "05-31"),
    "summer": ("06-01", "08-31"), 
    "fall":   ("09-01", "11-30"),
    "winter": ("12-01", "02-28"),  # Non-leap year for simplicity
}
TEST_SEASONS = False

TRAIN_START_YEAR = ''              # If left empty, will default to 2015 (Start of Dataset)
TRAIN_END_YEAR = '2018-12-31'
TEST_YEAR = 2019


# === Target and Feature Selection ===
TARGET_COL = 'SWC_10'
MANUAL_KEEP = ['SWC_10', 'T_5']

# === Tuning Mode ===
TUNE = False

# === Default Model Parameters ===
MODEL_PARAMS = {
    "cnn": {
        "filters": 64,
        "kernel_size": 3,
        "activation": "relu",
        "dense_units": 64,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
    },
    "lstm": {
        "lstm_units": 64,
        "activation": "tanh",
        "dense_units": 64,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
    },
    "arima": {
        "p": 1,
        "d": 1,
        "q": 1
    },
    "sarima": {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 24)
    },
    "autoarima": {},  # uses internal tuning
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1
    },
}

# === Hyperparameter Grids for Tuning ===

RAW_PARAM_GRID = {
    "cnn": {
        "filters": [32, 64],
        "kernel_size": [3, 5],
        "activation": ["tanh","relu"],
        "dense_units": [64, 128],
        "dropout_rate": [0.3, 0.5],
        "learning_rate": [0.001, 0.0005],
    },
    "lstm": {
        "lstm_units": [32, 64],
        "activation": ["tanh", "relu"],
        "dense_units": [64, 128],
        "dropout_rate": [0.3, 0.5],
        "learning_rate": [0.001, 0.0005],
    },
    "arima": {
        "p": [0, 1, 2],
        "d": [0, 1],
        "q": [0, 1, 2],
        "seasonal": [True, False]
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],        # Number of boosting rounds (trees)
        "max_depth": [3, 5, 7],                # Maximum depth of each tree
        "learning_rate": [0.01, 0.1, 0.3]
    }
}
# Automatically expand to full config list
PARAM_GRID = {
    model: list(ParameterGrid(grid))
    for model, grid in RAW_PARAM_GRID.items()
}

 
model_meta = {
    "cnn": {"requires_windowing": True},
    "lstm": {"requires_windowing": True},
    "arima": {"requires_windowing": False},
    "arimax": {"requires_windowing": False},
    "autoarima": {"requires_windowing": False},
    "sarima": {"requires_windowing": False},
    "xgboost": {"requires_windowing": True},
}

# "Ppt", "Tair", "RH", "Srad", "Wind_X", "Wind_Y"
EXOG_FEATURES = ['T_5', 'Ppt']
SEARCH_ARIMAX_FEATURES = False          # Whether to test all subsets of exogenous features

# Setup

How to use:
1. Run the following:
    ```bash
    pip install -e .
    ```
1. Under 'notebooks' feel free to run, copy, or use the "exploration.ipynb" file

# TODO
1. Data preprocessing
1. Analysis/Feature Engineering
    - Seasonal patterns
    - Feature Correlation
1. Model Trials
    - Establish a baseline with A/SARIMA
    - Compare with Random Forest/ XGBoost
1. Validation
    - Rolling/walk-forward validation
    - Compare performance metrics (MAE, RMSE, R^2) across all models.

# Documentation

## General Files

- `data/` - contains the data files used in the analysis
- `notebooks/` - Jupyter notebooks for data exploration
    - `exploration.ipynb` - The main driver notebook as of now
- `scripts/` - Auxiliary helper scripts
    - `tree_cmd.py` - Lets you run "make tree" in terminal and prints out file structure
    - `tree_ignore.txt` - Contains the files to ignore when running `tree_cmd.py`
- `results/` - Empty as of now

## Files with Relevant Code

- `src/` - Library of methods used in notebooks
    - `config/`
        - `config_loader.py` - Loads config file
        - `config.yaml` - Contains configurable constants
    - `data/`     
        - `data_helpers.py` - Methods for loading and saving data
        - `feature_engineering.py` - Add, remove, modify features
        - `windowing.py` - Creates windows as needed
    - `models`
        - `arima_helpers.py` - ARIMA/SARIMA model training and forecasting
        - `lstm_helpers.py` - LSTM model training and forecasting
    - `visualization`
        - `plotting.py` - Methods for plotting data

## Model Explanation

| **Model**  | **Type**  | **Best For**  | **Limitations** |
|------------|----------|--------------|----------------|
| **ARIMA**  | Statistical | Short-term, stationary time series | Struggles with seasonality, requires manual differencing |
| **SARIMA** | Statistical (Seasonal ARIMA) | Time series with strong seasonal patterns | Slow for large datasets, requires seasonal tuning |
| **LSTM**   | Deep Learning | Large, complex time series with long-term dependencies | Needs more data, longer training time |

- **ARIMA** is implemented in `arima_helpers.py` and is best for **short-term predictions** on stationary data.
- **SARIMA** extends ARIMA with **seasonality (`m=365` for yearly cycles)**.
- **LSTM** in `lstm_helpers.py` is a **neural network model** that learns patterns **automatically** and works well for **large datasets**.


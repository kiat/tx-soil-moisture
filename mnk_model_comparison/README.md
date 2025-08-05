# Soil Moisture Model Comparison Pipeline

This repository contains a machine learning pipeline for comparing multiple forecasting models (CNN, LSTM, ARIMA, XGBoost) on soil moisture and related datasets. The pipeline supports hyperparameter tuning, feature selection, and evaluation across different forecast horizons and seasonal splits.

---

## Repository Structure

- **`config.py`** – Central configuration for model parameters, feature selection, forecast horizons, and boolean flags controlling seasonal tests and other options.
- **`helpers.py`** – Utility functions for data preprocessing, feature engineering, evaluation metrics, and plotting.
- **`model_runner.py`** – Main entry point for running the full pipeline, either from a Jupyter Notebook or the command line. Handles training, testing, and logging.
- **`imports.py`** – A single file containing all required imports for the project. Importing this file smoothly sets up your environment.
- **`data_analysis.ipynb`** – Jupyter notebook for exploratory data analysis (EDA) and interactive testing of the pipeline.
- **`evaluation_results.csv`** – Output file containing results for each model/config combination.

---

## Using `imports.py`

Instead of manually importing all required packages, you can load them with:
```python
from imports import *
```
This will bring in all system, data, ML, and deep learning imports.

helpers.py, config.py, and model_runeer.py will need manual import.

---

## Running from a Notebook

Example:
```python
from imports import *
data = pd.read_csv("datasets/soil_data.csv")
run_pipeline(data, models=['cnn', 'lstm', 'arima', 'xgboost'])
```

---

## Output
The pipeline logs results to `evaluation_results.csv` after each model configuration is run.  
Example output columns:
```
model,params,output_horizon,season,rmse,mae,mape,corr,features
lstm,"{'lstm_units': 32, ...}",1,All,0.234,0.210,12.5,0.89,['SWC_10','T_5',...]
```

## Configuration (`config.py`)

The `config.py` file contains all tunable parameters, file paths, and flags that control the behavior of the pipeline.
You can modify these values either directly in the file, or dynamically when running from a Jupyter Notebook or the command line.

### Key Flags and Variables

- **DATA_PATH**: Path to your dataset.
- **TARGET_COL**: Name of the target column for prediction.
- **TEST_SEASONS**: If `True`, runs the pipeline separately for each season.
- **TEST_ALL**: If `True`, runs the pipeline separately for each output horizon provided.
- **RAW_PARAM_GRID**: Dictionary defining the hyperparameter search space for each model.
- **OUTPUT_HORIZONS**: List of forecast horizons to evaluate (e.g., `[1, 3, 5]`).
- **MANUAL_KEEP**: Feature names to always keep during feature selection.
- **EXOG_FEATURES**: Exogenous variables for ARIMAX/SARIMAX models.
---

## Running from the Command Line

You can execute the pipeline directly from the terminal:
```bash
python model_runner.py --data_path path/to/your_dataset.csv --models cnn lstm arima xgboost
```

### Available Flags
| Flag                | Type        | Description |
|---------------------|-------------|-------------|
| `--data_path`       | str         | Path to CSV dataset. |
| `--models`          | list[str]   | Models to run (`cnn`, `lstm`, `arima`, `xgboost`). |
| `--horizons`        | list[int]   | Forecast horizons to evaluate. |
| `--test_seasons`    | bool        | Whether to run seasonal splits (True/False). |
| `--normalize`       | bool        | Apply normalization to features. |
| `--manual_features` | bool        | Use manually selected features instead of auto feature selection. |
| `--log_path`        | str         | Output CSV path for results. |

---

## Output

After each model/configuration is tested, results are appended to `evaluation_results.csv` with the following columns:

- `model` – Model name
- `params` – Hyperparameters used
- `output_horizon` – Forecast horizon in days
- `season` – Season name or "All"
- `rmse`, `mae`, `mape`, `corr` – Evaluation metrics
- `features` – Features used for training

---

## Example Workflow

From Notebook:
```python
from imports import *
from model_runner import run_pipeline
data = pd.read_csv("datasets/soil_data.csv")
run_pipeline(data, models=['cnn', 'xgboost'])
```


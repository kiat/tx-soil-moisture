# Soil Moisture Data Imputation Pipeline

#### Zun

This project provides a multi-stage pipeline to clean raw soil moisture sensor data and fill in missing values. The pipeline uses a sequence of increasingly sophisticated methods, chosen based on the length of each data gap.

## Pipeline Workflow

The process is orchestrated by `imputation_pipeline.py`, which runs the following scripts in order:

1.  **`datacleaning.py`**:
    *   **Input**: Raw soil (`.dat`) and meteorological (`.dat`) data files.
    *   **Action**: Merges datasets, validates data against physical bounds (e.g., SWC 0-0.6), and replaces invalid values with `NaN`.
    *   **Output**:
        *   `cleaned_data/Station{id}_cleaned_data.csv`: A clean, hourly-indexed timeseries.
        *   `missing_data/Station{id}_missing_data.csv`: A log of all identified gaps.

2.  **`Shortgaps.py`**:
    *   **Input**: `..._cleaned_data.csv`
    *   **Action**: Fills gaps shorter than **24 hours** using **monotonic cubic interpolation (`pchip`)**. This method is chosen for short gaps because it is computationally fast and creates a smooth, visually plausible curve that honors the data points at the edges of the gap. Unlike standard spline interpolation, `pchip` is shape-preserving, meaning it will not "overshoot" and create artificial peaks or troughs in the data.
    *   **Output**: `output/Station{id}_filled_shortgaps.csv`

3.  **`Mediumgaps.py`**:
    *   **Input**: `..._filled_shortgaps.csv`
    *   **Action**: Fills gaps between **24 and 168 hours** (1-7 days) using a **SARIMAX** (Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors) model. This advanced statistical model is ideal for time series with strong seasonality, such as the 24-hour diurnal cycle seen in soil moisture. It makes predictions based on past values, past errors, and external variables. To improve accuracy, the model uses **precipitation (`Ppt`) as a key external variable**. If a station's own precipitation data is missing, the script intelligently uses data from Station 3 as a reliable fallback. The script then uses `auto_arima` to automatically find the optimal model parameters, ensuring a robust forecast for each specific gap.
    *   **Output**: `output/Station{id}_filled_mediumgaps.csv`

4.  **`Longgaps.py`**:
    *   **Input**: `..._filled_mediumgaps.csv`
    *   **Action**: Fills gaps between **7 and 30 days** using an **XGBoost machine learning model**. For longer gaps where statistical time-series models weaken, this script builds a rich set of predictive features, including lagged values, rolling averages, and time-based indicators (hour of day, day of year). It trains the powerful XGBoost model on these features to learn complex patterns and predict the missing values. A linear "drift correction" is applied to the predictions to ensure a seamless and smooth transition from the real data into and out of the filled gap.
    *   **Output**: `output/Station{id}_filled_longgaps.csv`

5.  **`VeryLongGaps.py`**:
    *   **Input**: `..._filled_longgaps.csv`
    *   **Action**: Fills gaps of **30 days or more** using **cross-station linear regression**. When a station's own data is missing for an extended period, this script looks for help from its neighbors. It systematically finds the most highly correlated "donor" station that has complete data during the gap. It then trains a simple linear regression model based on the historical relationship between the two stations and uses this model to "translate" the donor station's data to fill the gap in the target station.
    *   **Output**: `output/Station{id}_filled_verylonggaps.csv`

## Requirements

- Python 3.7+
- Required packages are listed in `requirements.txt`.

Install all dependencies with a single command:
```bash
pip install -r requirements.txt
```

## How to Run

It is highly recommended to use the main pipeline script, which handles the entire workflow automatically.

### Automated Pipeline (Recommended)

The `imputation_pipeline.py` script is the master runner. Navigate to the `data-cleanup` directory and run it.

**Run the entire pipeline for all stations (1-6):**
```bash
python imputation_pipeline.py
```

**Run for a single station:**
```bash
python imputation_pipeline.py --station 3
```

**Perform a "dry run" to see the commands without executing them:**
```bash
python imputation_pipeline.py --dry
```

### Manual Step-by-Step Execution (Advanced)

If you need to run a specific part of the pipeline for debugging or analysis, you can execute each script individually. Ensure you run them in the correct order as listed in the "Workflow" section above.

**Example: Manually running the first two steps for Station 1**

1.  **Run the initial data cleaning:**
    ```bash
    python datacleaning.py --station 1
    ```
    *(This creates `cleaned_data/Station1_cleaned_data.csv` and the missing data log)*

2.  **Run the short gap filling:**
    ```bash
    python Shortgaps.py --station 1
    ```
    *(This reads the file from step 1 and creates `output/Station1_filled_shortgaps.csv`)*

You can continue this pattern for the remaining scripts. Each script includes `--help` for more details on its arguments.
```bash
python Mediumgaps.py --help
```
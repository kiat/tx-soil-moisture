# Running Scripts

## convert_missing_data_to_NaN.py

The script ```convert_missing_data_to_NaN.py``` inputs the original data folder files, so one SWC file and one MET file for each station. The script identifies missing and invalid values in the dataset, and marks those values as NaN values. The ```missing_cleaned_data``` directory is created automatically to hold the output file for each station. You can run a single station at a time, or run the flag ```--all``` to run all stations at once. Examples of terminal commands and working outputs are provided. Here is the output of the script: 

- ```cleaned_data_station_1.csv```
  - This is a csv file of a cleaned dataset that combines both  MET and SWC datasets, with invalid and missing values removed and marked as NaN.


### How to run convert_missing_data_to_NaN.py:

Add environment variables in terminal first.
```
export SOIL_DATA_DIR=/path/to/soil_station
export MET_DATA_DIR=/path/to/met_station
```

Then, you can run the following. You can replace the # with the number of the specific station you wish to clean, or use the ```--all``` flag to run all stations.
Example usage:
```
python3 convert_missing_data_to_NaN.py --station #           # processes data for station only
python3 convert_missing_data_to_NaN.py --all                 # processes data for all stations 
```

The final output in the terminal should look like this, this example covers stations 1 to 6:

```
Processing Station 1...
Cleaned data saved to: missing_cleaned_data/cleaned_data_station_1.csv
Processing Station 2...
Cleaned data saved to: missing_cleaned_data/cleaned_data_station_2.csv
Processing Station 3...
Cleaned data saved to: missing_cleaned_data/cleaned_data_station_3.csv
Processing Station 4...
Cleaned data saved to: missing_cleaned_data/cleaned_data_station_4.csv
Processing Station 5...
Cleaned data saved to: missing_cleaned_data/cleaned_data_station_5.csv
Processing Station 6...
Cleaned data saved to: missing_cleaned_data/cleaned_data_station_6.csv
```

****************************************************************
Spring 2025 Data Cleanup Team: Abi, Nethra, Ramya, Zun


# Soil Data Pipeline - datacleaning


## Features

1. **Load and parse raw data**  
   - Soil data (`SM_{station}.dat`) and MET data (`MET_{station}.dat`) are read from configurable directories.  
   - Dates are converted to a `DateTimeIndex` and numeric columns are coerced to floats (invalid parse → NaN).

2. **Merge datasets**  
   - Soil and MET data are joined on the timestamp index (inner join).  
   - When both stations report precipitation (`Ppt_soil` & `Ppt_met`), the MET value is kept and soil’s is dropped.

3. **Validate and clean**  
   - **Missing data**: scans each column for NaNs and records their timestamps.  
   - **Invalid data**: applies realistic bounds to physical measurements, replacing out-of-range values with NaN:  
     - Soil moisture (`SWC_*`): 0–0.6  
     - Precipitation (`Ppt`): ≥ 0  
     - Relative humidity (`RH`): 0–100%  
     - Wind speed: 0–25 m/s  
     - Wind direction: 0–360°  
     - Solar radiation (`Srad`): ≥ 0  
     - Temperature (`T_*`, `Tair`): –30–60 °C

4. **Generate summaries and outputs**  
   - **Raw merged data** saved to `raw_merged_data/raw_merged_station_{station}.csv`.  
   - **Missing/invalid summary** grouped by consecutive timestamp runs, exported to `missing_data/Station{station}_missing_data.csv` or custom path.  
   - **Cleaned full-timeline data** reindexed to an hourly grid (from first to last timestamp), with all invalid/missing replaced by NaN, saved to `cleaned_data/Station{station}_cleaned_data.csv` or custom path.

## Requirements

- Python 3.7 or higher  
- `pandas`  
- `numpy`

Install dependencies via:

```bash
pip install pandas numpy
```

## Usage

```bash
python datacleaning.py --station 1
```

**Optional arguments:**

- `-s`, `--station` (int, default=1): Station ID (1–6)  
- `--soil-base-dir` (str): Path to soil `.dat` files (default `../datasets/TX-Data/soil_station`)  
- `--met-base-dir` (str): Path to MET `.dat` files (default `../datasets/TX-Data/met_station`)  
- `--raw-output-dir` (str): Directory for raw merged CSVs (default `raw_merged_data`)  
- `--missing-output` (str): Filename for missing/invalid summary CSV (default `missing_data/Station{station}_missing_data.csv`)  
- `--cleaned-output` (str): Filename for cleaned full-timeline CSV (default `cleaned_data/Station{station}_cleaned_data.csv`)

### Batch processing example

You can run multiple stations in sequence:

```bash
python datacleaning.py --station 1
python datacleaning.py --station 2
python datacleaning.py --station 3
python datacleaning.py --station 4
python datacleaning.py --station 5
python datacleaning.py --station 6
...
```




# Soil Moisture Data Imputation Pipeline

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
    *   **Action**: Fills gaps shorter than **24 hours** using monotonic cubic interpolation (`pchip`).
    *   **Output**: `output/Station{id}_filled_shortgaps.csv`

3.  **`Mediumgaps.py`**:
    *   **Input**: `..._filled_shortgaps.csv`
    *   **Action**: Fills gaps between **24 and 168 hours** (1-7 days) using a `SARIMAX` time series model, which accounts for daily seasonality.
    *   **Output**: `output/Station{id}_filled_mediumgaps.csv`

4.  **`Longgaps.py`**:
    *   **Input**: `..._filled_mediumgaps.csv`
    *   **Action**: Fills gaps between **7 and 30 days** using an `XGBoost` machine learning model with rich time-based and environmental features.
    *   **Output**: `output/Station{id}_filled_longgaps.csv`

5.  **`VeryLongGaps.py`**:
    *   **Input**: `..._filled_longgaps.csv`
    *   **Action**: Fills gaps of **30 days or more** using cross-station linear regression, finding the most correlated "donor" station to model the missing data.
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
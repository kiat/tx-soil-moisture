# CYGNESS 

https://cygnss.engin.umich.edu/


# SMAP Satellite 

It is better to use SMAP Satellite 

https://smap.jpl.nasa.gov/


# Merge.py & Modify.py

These scripts are responsible for data integration and modification:

Merge.py:
- Combines AMSR data with SMAP satellite data
- Integrates the satellite data with simulated data
- Ensures proper alignment of timestamps and data points across different sources
- Handles any discrepancies in data formats or units

Modify.py:
- Performs necessary modifications on the merged dataset
- May include data cleaning, formatting, or additional preprocessing steps
- Prepares the data for further analysis and modeling

Together, these scripts create a comprehensive dataset that combines multiple sources of soil moisture data, providing a robust foundation for subsequent analysis and modeling tasks.

MetMerge.py:
- Combines AMSR data with SMAP satellite data
- Keeps out the SWC data

# Autoregressive_LSTM.py & run_all_combinations.sh

These scripts are used for running LSTM and autoregressive models on the satellite data:

Autoregressive_LSTM.py:
- Implements both LSTM and autoregressive neural network models
- Evaluates feature importance for soil moisture prediction
- Supports configurable parameters:
  - Stations (combines data from selected stations, e.g., "Station1 Station2" uses data from both stations together)
  - Target (choose one: Sat_SM_AMSR or Sat_SM_SMAP)
  - Model type (choose one: lstm or autoregressive)
  - Number of steps for prediction

run_all_combinations.sh:
- Shell script that runs ALL possible combinations:
  - Every possible combination of stations (1-6 stations), analyzing their combined data
    * Single station data
    * Combined data from pairs of stations
    * Combined data from triplets of stations
    * And so on up to combined data from all six stations
  - Both targets (AMSR and SMAP), one at a time
  - Both model types (lstm and autoregressive), one at a time
  - All steps from 1 through 7
- Creates a LSTM_results directory to store outputs
- Generates detailed CSV files with results sorted by MAE percentage

## How to Run

1. Make the shell script executable:

```bash
chmod +x satellite/run_all_combinations.sh
```

2. Run either a single configuration or all combinations:

### Single Configuration (Shortest):

```bash
python satellite/Autoregressive_LSTM.py \
    --stations "Station1" \
    --target "Sat_SM_AMSR" \
    --model_type "lstm" \
    --steps 1
```

### All Possible Combinations:

```bash
./satellite/run_all_combinations.sh
```
This will run:
- All possible station data combinations (63 total):
  * Individual station data
  * Combined data from 2 stations
  * Combined data from 3 stations
  * Combined data from 4 stations
  * Combined data from 5 stations
  * Combined data from all 6 stations
- Both targets (AMSR and SMAP)
- Both model types (lstm and autoregressive)
- Seven step values (1 through 7)
Total: 63 * 2 * 2 * 7 = 1,764 different configurations

## Example Configurations

1. Minimal Test (Single Station):

```bash
python satellite/Autoregressive_LSTM.py \
    --stations "Station1" \
    --target "Sat_SM_AMSR" \
    --model_type "lstm" \
    --steps 1
```

2. Multiple Stations (Combined Analysis):

```bash
python satellite/Autoregressive_LSTM.py \
    --stations "Station1" "Station2" "Station3" \
    --target "Sat_SM_SMAP" \
    --model_type "lstm" \
    --steps 7
```
This will analyze the combined data from Stations 1, 2, and 3 together.

3. Full Analysis:

```bash
./satellite/run_all_combinations.sh
```
This will test all possible combinations of station data, analyzing each combination's data together.

Results will be saved in the satellite/LSTM_results directory, with filenames indicating the configuration used. Results are sorted by MAE percentage, with the best performing models listed first.

# Cleaning_and_Analysis.ipynb

This Jupyter notebook focuses on data cleaning and analysis for soil moisture and meteorological data. Key steps include:

- Importing necessary libraries (pandas, numpy, matplotlib, etc.)
- Reading and preprocessing data from multiple stations
- Merging soil moisture and meteorological data
- Cleaning the data by removing spaces in column names, converting data types, and handling missing values
- Creating visualizations to explore the data
- Performing feature engineering, such as calculating moving averages
- Normalizing the data
- Saving the cleaned and processed data to CSV files

# Data_Exploration.ipynb

This notebook is dedicated to exploring the cleaned and merged data. It includes:

- Loading the preprocessed data
- Generating descriptive statistics
- Creating various visualizations to understand the relationships between different features
- Analyzing correlations between variables
- Exploring time series patterns in the data

# model_comparison.py

This Python script focuses on evaluating feature importance for different models, particularly RNN and CNN. Key components include:

- Importing required libraries and modules
- Loading and preprocessing the data
- Defining functions for feature importance calculation
- Implementing RNN and CNN models for feature importance evaluation
- Calculating and displaying feature importance scores for different metrics (MSE, MAE, MAE%, R²)
- Handling exceptions and errors in the evaluation process
- Saving results to CSV files


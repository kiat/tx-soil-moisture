# Interpolation Strategy for Missing Data

We classify missing data based on the gap length:

## Short Gaps (Daily missing)
- **Method**: 
  - **KNN (K-Nearest Neighbors)** (using scikit-learn)
    - Non-parametric supervised learning algorithm
    - Predicts based on the nearest values beside it (good for short term gaps)
    - Use 24 hours before and after the missing point
- **Variables**:
  - **SWC, SWC_5-50**: KNN or linear interpolation
  - **Tair**: Use nearby station values
  - **RH (Relative Humidity)**: Use inverse relationship with Tair to check consistency
  - **Wind speed**: Basic interpolation
  - **Wind direction**: Circular interpolation may be needed
  - **Srad**: Use 24 near neighbors before and after

## Medium Gaps (Weekly missing)
- **Methods**:
  - **SARIMA** or **ARIMA**: accounts for seasonality
  - **Regression-based models**: Random Forest, Gradient Boosting
    - Can use unless *all features* are NaN
  - **Holt-Winters**: 
    - Good for seasonality, cyclic behavior, and smoothing averages
- **Variables**:
  - **Tair**: Nearby station values
  - **RH**: Same as short gaps (inverse with Tair)
  - **Wind speed**: Standard interpolation
  - **Wind direction**: Circular interpolation
  - **Srad**: Interpolate considering seasonal patterns
  - **Ppt (Precipitation)**: 
    - Use nearby station values
    - **Idea**: First classify into "yes it rained" and "no it didn’t rain"
    - Then, if "yes", predict the rain amount

## Long Gaps (1 week - 1 month)
- **Methods**:
  - **Prophet** (from Facebook/Meta)
    - Designed for time series with strong seasonal effects
    - Robust to missing data and seasonality changes
  - **State-Space Models**
- **Variables**: Same general strategy as medium gaps

## Very Long Gaps (Longer than 1 month)
- **Methods**:
  - **Machine Learning algorithms**:
    - Random Forest
    - Gradient Boosting
    - LSTM (Long Short-Term Memory neural networks)


# Running Scripts

## Script 1: merges MET and SOIL data

### How to run script1.py:

Add environment variables first in terminal
```
export SOIL_DATA_DIR=/path/to/soil_station
export MET_DATA_DIR=/path/to/met_station
```

Then 
```
python script1.py #
```

Example usage:
```
python script1.py 1  // creates merged file for station_1 
python script1.py all // creates merged files for all stations
```

Output: creates a file called ```raw_merged_data/raw_merged_station_1.csv```
```
Saved cleaned data to: raw_merged_data/raw_merged_station_1.csv
```


## Script 2: outputs 3 files
- List of individual timestamps of which variable is missing for that station
- Timestamp (range) summary of missing data for that station
- Cleaned dataset replacing missing & invalid data

### How to run script2.py:
```
python script2.py --station #
```

Example usage:
```
python script2.py --station # // three files outputted for station 1
python script2.py --station all // all stations processed
```

Output: creates three files called ```missing_cleaned_data/Station1_missing_timestamps.csv```, ```missing_cleaned_data/Station1_missing_data.csv```, and ```missing_cleaned_data/Station1_cleaned_data.csv```
```
Individual Missing Timestamps saved to: missing_cleaned_data/Station1_missing_timestamps.csv
Missing data summary saved to: missing_cleaned_data/Station1_missing_data.csv
Cleaned data saved to: missing_cleaned_data/Station1_cleaned_data.csv
```


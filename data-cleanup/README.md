# Running Scripts

## Script 1

Script 1 merges the SWC and MET files while removing invalid values. This script outputs four files, each of which are explained below. The ```missing_cleaned_data``` directory is created automatically to hold the output file.
- ```cleaned_data_station_1.csv```
  - A cleaned dataset that combines both  MET and SWC datasets, with invalid values removed. The missing data and individual missing timestamps were found by using this dataset 


### How to run cleaning_script.py:

Add environment variables in terminal first.
```
export SOIL_DATA_DIR=/path/to/soil_station
export MET_DATA_DIR=/path/to/met_station
```

Then, you can run the following.

``` python3 script1.py --station # ```: station numbers in place of #, or "all"

Example usage:
```
python3 script1.py --all                 # processes data for all stations 
python3 script1.py --station 1           # processes data for station_1 only
```

The final output in the terminal should look like this:

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


## Script 2

Script is still under progress.


****************************************************************
Spring 2025 Data Cleanup Team: Abi, Nethra, Ramya, Zun

TO-DO:
- Classify short, medium, long, and very large gaps, with an assessment of using Station 3 (which has complete data) as a reference.  1. Short Gaps (Hours to Days) (<24 hours) 2. Medium Gaps (Days to Weeks) (1–7 days) (24–168 hours) 3. Long Gaps (Weeks to Months) (7–30 days) (168 hours–720 hours) 4. Very Large Gaps (Months to Years) (over 30 days) (>720 hours)
- Overal Data Imputation Script

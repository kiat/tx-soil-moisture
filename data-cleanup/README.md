# Running Scripts

## convert_missing_data_to_NaN.py

The script ```convert_missing_data_to_NaN.py``` inputs the original data folder files, so one SWC file and one MET file for each station. The script identifies missing and invalid values in the dataset, and marks those values as NaN values. The ```missing_cleaned_data``` directory is created automatically to hold the output file for each station. You can run a single station at a time, or run the flag ```--all``` to run all stations at once. Examples of terminal commands and working outputs are provided. Here is the output of the script: 

- ```cleaned_data_station_1.csv```
  - This is a csv file of a cleaned dataset that combines both  MET and SWC datasets, with invalid and missing values removed and marked as NaN.


### How to run con.py:

Add environment variables in terminal first.
```
export SOIL_DATA_DIR=/path/to/soil_station
export MET_DATA_DIR=/path/to/met_station
```

Then, you can run the following. You can replace the # with the number of the specific station you wish to clean, or use the ```--all``` flag to run all stations
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

TO-DO:
- Classify short, medium, long, and very large gaps, with an assessment of using Station 3 (which has complete data) as a reference.  1. Short Gaps (Hours to Days) (<24 hours) 2. Medium Gaps (Days to Weeks) (1–7 days) (24–168 hours) 3. Long Gaps (Weeks to Months) (7–30 days) (168 hours–720 hours) 4. Very Large Gaps (Months to Years) (over 30 days) (>720 hours)
- Overal Data Imputation Script

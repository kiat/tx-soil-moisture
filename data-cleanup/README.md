



# Running Scripts

## Script 1

Script 1 merges the SWC and MET files while removing invalid values. This script outputs four files, each of which are explained below. Two directories, ```raw_merged_data``` and ```missing_cleaned_data```, are created automatically to hold the output files.
- ```raw_merged_station_1.csv```
  - A cleaned dataset that combines both original MET and SWC datasets, no changes to actual values.
- ```missing_ranges_station_1.csv```
  - A list recording the parameter, start timestamp, end timestamp, and number or missing data values.
- ```missing_timestamps_station_1.csv```
  - A list recording the timestamp (by hour) and parameter that is missing
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
Raw merged data saved to: raw_merged_data/raw_merged_station_1.csv
Missing data ranges saved to: missing_cleaned_data/missing_ranges_station_1.csv
Individual missing timestamps saved to: missing_cleaned_data/missing_timestamps_station_1.csv
Cleaned data saved to: missing_cleaned_data/cleaned_data_station_1.csv
```

#### Deleting Files

When you need to delete files, you can either delete the files for one specific station or all the stations
Example usage:

```
python3 script1.py --all --clean         # deletes .csv files for all stations
python3 script1.py --station 1 --clean   # deletes .csv files for station 1 only
```

This is the out that is supposed to appear in the terminal:

```
Deleted files for Station 1 in raw_merged_data: ['raw_merged_station_1.csv']
Deleted files for Station 1 in missing_cleaned_data: ['cleaned_data_station_1.csv', 'missing_ranges_station_1.csv', 'missing_timestamps_station_1.csv']
```

## Script 2

Script is still under progress.


****************************************************************
Spring 2025 Data Cleanup Team: Abi, Nethra, Ramya, Zun

TO-DO:
- Classify short, medium, long, and very large gaps, with an assessment of using Station 3 (which has complete data) as a reference.  1. Short Gaps (Hours to Days) (<24 hours) 2. Medium Gaps (Days to Weeks) (1–7 days) (24–168 hours) 3. Long Gaps (Weeks to Months) (7–30 days) (168 hours–720 hours) 4. Very Large Gaps (Months to Years) (over 30 days) (>720 hours)
- Overal Data Imputation Script

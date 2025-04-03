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
```Saved cleaned data to: raw_merged_data/raw_merged_station_1.csv```

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
python script2.py --station all // all station processed
```

Output: creates three files called ```missing_cleaned_data/Station1_missing_timestamps.csv```, ```missing_cleaned_data/Station1_missing_data.csv```, ```missing_cleaned_data/Station1_cleaned_data.csv```
```
Individual Missing Timestamps saved to: missing_cleaned_data/Station1_missing_timestamps.csv
Missing data summary saved to: missing_cleaned_data/Station1_missing_data.csv
Cleaned data saved to: missing_cleaned_data/Station1_cleaned_data.csv
```


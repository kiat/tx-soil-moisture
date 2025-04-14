



# Running Scripts

## Script 1: Cleans Data

Script 1 merges the SWC and MET files while removing invalid values. This script outputs four files, each of which are explained below.
- ```raw_merged_data/raw_merged_station_1.csv```
  - A cleaned dataset that combines both original MET and SWC datasets, no changes to actual values.
-  ```missing_cleaned_data/Station1_missing_data.csv```
  - A list recording the parameter, start timestamp, end timestamp, and number or missing data values.
- ```missing_cleaned_data/Station1_missing_timestamps.csv```
  - A list recording the timestamp (by hour) and parameter that is missing
- ```missing_cleaned_data/Station1_cleaned_data.csv```
  - A cleaned dataset that combines both  MET and SWC datasets, with invalid values removed. The missing data and individual missing timestamps were found by using this dataset/ 


### How to run cleaning_script.py:

Add environment variables first in terminal
```
export SOIL_DATA_DIR=/path/to/soil_station
export MET_DATA_DIR=/path/to/met_station
```

Then 
```
python3 cleaning_script.py --station #
```

Example usage:
```
python3 cleaning_script.py --station 1 // processes data station_1 
python3 cleaning_script.py --station all // processes data for all stations
```

The final output should look like this, while the ```raw_merged_data``` and ```missing_cleaned_data``` directories are created to hold the output files.

```
Raw merged data saved to: raw_merged_data/raw_merged_station_1.csv
Missing data summary saved to: missing_cleaned_data/Station1_missing_data.csv
Individual missing timestamps saved to: missing_cleaned_data/Station1_missing_timestamps.csv
Cleaned data saved to: missing_cleaned_data/Station1_cleaned_data.csv
```



****************************************************************
Spring 2025 Data Cleanup Team: Abi, Nethra, Ramya, Zun

Here is the deepnote link for our working notebook:
https://deepnote.com/workspace/UT-Austin-fe53bdd3-3bba-4d00-8ae5-a7ded30c81d4/project/2025SoilMoistureProject-25f43899-5656-4a70-8169-abc0aacffbd3/notebook/Notebook-1-b3dbdcbd31604f5599e27d7bdd9f4c09?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=25f43899-5656-4a70-8169-abc0aacffbd3


Finish putting the script into a format where we give a file input and recieve a file output
NEW FEATURE: Be able to query for the following statistics...
- Ask for soil moisture value for each month for each station (should complete a PEMDAS operation)
- Ask for soil moisture value for each moth for each season
- Ask for soil moisture value average for each month each station per year
- ^ Use the soil moisture value average to find what the model would be if we replace the predicted value w calculated avg prediction of the month (bcz the avg value is between 0-0.6, January is usually between 0-0.3, and august is usually 0-0.5)


TO-DO:
- Classify short, medium, long, and very large gaps, with an assessment of using Station 3 (which has complete data) as a reference.  1. Short Gaps (Hours to Days) (<24 hours) 2. Medium Gaps (Days to Weeks) (1–7 days) (24–168 hours) 3. Long Gaps (Weeks to Months) (7–30 days) (168 hours–720 hours) 4. Very Large Gaps (Months to Years) (over 30 days) (>720 hours)
- Overal Data Imputation Script


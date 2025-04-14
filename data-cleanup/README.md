



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
python3 cleaning_script.py --station #
```

Example usage:
```
python3 cleaning_script.py --station 1 // processes data station_1 
python3 cleaning_script.py --station all // processes data for all stations
```

Output: creates a four files called ```raw_merged_data/raw_merged_station_1.csv```, ```missing_cleaned_data/Station1_missing_timestamps.csv```, ```missing_cleaned_data/Station1_missing_data.csv```, and ```missing_cleaned_data/Station1_cleaned_data.csv```
```
Raw merged data saved to: raw_merged_data/raw_merged_station_1.csv
Missing data summary saved to: missing_cleaned_data/Station1_missing_data.csv
Individual missing timestamps saved to: missing_cleaned_data/Station1_missing_timestamps.csv
Cleaned data saved to: missing_cleaned_data/Station1_cleaned_data.csv
```

Spring 2025 Data Cleanup Team: Abi, Nethra, Ramya, Zun

Here is the deepnote link for our working notebook:
https://deepnote.com/workspace/UT-Austin-fe53bdd3-3bba-4d00-8ae5-a7ded30c81d4/project/2025SoilMoistureProject-25f43899-5656-4a70-8169-abc0aacffbd3/notebook/Notebook-1-b3dbdcbd31604f5599e27d7bdd9f4c09?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=25f43899-5656-4a70-8169-abc0aacffbd3

- Overall, script is ready, needs to print out cleaned Station files
- Need to collaborate to include Zun's working script & data visualizations


Finish putting the script into a format where we give a file input and recieve a file output
NEW FEATURE: Be able to query for the following statistics...
- Ask for soil moisture value for each month for each station (should complete a PEMDAS operation)
- Ask for soil moisture value for each moth for each season
- Ask for soil moisture value average for each month each station per year
- ^ Use the soil moisture value average to find what the model would be if we replace the predicted value w calculated avg prediction of the month (bcz the avg value is between 0-0.6, January is usually between 0-0.3, and august is usually 0-0.5)

April 24th updates:
Have a final Script 1 done!

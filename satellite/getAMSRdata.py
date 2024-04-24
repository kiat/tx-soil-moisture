import os
import glob
from datetime import datetime

# Set the directory path where the uploaded file is located
directory_path = '/Users/jamisenma/tx-soil-moisture/satellite/'

# Create the pattern for the files you want to list
pattern = 'AMSR_U2_L2_Land_B02*'

# Use glob.glob() to list all files that match the pattern
all_files = glob.glob(os.path.join(directory_path, pattern))

# This will hold the selected file for each day
selected_files = {}

# Go through each file, parse the date, and select one file per day
for file in all_files:
    # Extract the date part from the filename
    # Assuming the format "AMSR_U2_L2_Land_B02_YYYYMMDDxxxx_D.he5"
    print(file)
    print("bruh")
    date_part = os.path.basename(file)[20:28]


    # Debug: print the date_part to make sure it's correct
    print("Date part extracted:", date_part)

    # Parse the date
    try:
        date = datetime.strptime(date_part, "%Y%m%d").date()
    except ValueError as e:
        print("Error parsing date from filename:", file)
        print(e)
        continue  # Skip this file

    # If we haven't seen this date, add the file to the selected list
    if date not in selected_files:
        selected_files[date] = file

# Now we have one file per day
selected_files_per_day = list(selected_files.values())

files = sorted(selected_files_per_day)

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import math


def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Difference in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Earth's radius in kilometers (can be adjusted for different units)
    R = 6371.0
    distance = R * c

    return distance


def unix_to_datetime(unix_timestamp):
    # Converts a Unix timestamp to a human-readable datetime in UTC
    return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')


def find_closest_data_point(station, df):
    # Finds the closest data point to a given station
    df = df.reset_index(drop=True)
    distances = df.apply(
        lambda row: haversine(station['Latitude'], station['Longitude'], row['Latitude'], row['Longitude']), axis=1)
    min_index = distances.idxmin()
    min_distance = distances[min_index]
    closest_point = df.iloc[min_index]
    return closest_point, min_distance


def read_soil_moisture(file_path):
    # Define your stations
    stations_data = [
        {'Latitude': 30.3989, 'Longitude': -98.6105, 'Station_ID': '5'},
        {'Latitude': 30.4193, 'Longitude': -98.8046, 'Station_ID': '1'},
        {'Latitude': 30.4421, 'Longitude': -98.8427, 'Station_ID': '6'},
        {'Latitude': 30.4600, 'Longitude': -98.9407, 'Station_ID': '4'},
        {'Latitude': 30.2454, 'Longitude': -98.7059, 'Station_ID': '2'},
        {'Latitude': 30.2758, 'Longitude': -98.7242, 'Station_ID': '3'}
    ]
    stations_df = pd.DataFrame(stations_data)

    with h5py.File(file_path, 'r') as file:
        # Navigate to the dataset containing soil moisture data
        dataset_path = 'HDFEOS/POINTS/AMSR-2 Level 2 Land Data/Data/Combined NPD and SCA Output Fields'
        dataset = file[dataset_path]

        # Read the data
        data = dataset[:]

        # Convert the structured array to a DataFrame
        full_df = pd.DataFrame(data)

        # Select only the required columns
        selected_columns = ['Time', 'Latitude', 'Longitude', 'SoilMoistureNPD']
        df_selected = full_df[selected_columns]

        # Convert Unix timestamps in the 'Time' column to human-readable datetimes
        df_selected['Time'] = df_selected['Time'].apply(unix_to_datetime)

        # Filter out rows where SoilMoistureNPD is -9999.0
        df_filtered = df_selected[df_selected['SoilMoistureNPD'] != -9999.0]

    closest_points_list = []

    for _, station in stations_df.iterrows():
        closest_point, distance = find_closest_data_point(station, df_filtered)
        closest_point['Station_ID'] = station['Station_ID']  # Add the Station_ID to the closest data point
        closest_point['distance'] = distance
        closest_points_list.append(closest_point)

    closest_points = pd.concat(closest_points_list, axis=1).transpose().reset_index(drop=True)

    return closest_points


full_dataframe = pd.DataFrame()

for file in files:
    # Read the closest data points for the current file
    closest_data_points = read_soil_moisture(file)
    print(closest_data_points)

    # Concatenate the closest data points to the full dataframe
    full_dataframe = pd.concat([full_dataframe, closest_data_points])

# After the loop, print out the full dataframe
for station_id, group in full_dataframe.groupby('Station_ID'):
    file_name = f'ASMR_station_{station_id}_data.csv'
    group.to_csv(file_name, index=False)
    print(f"Data for Station {station_id} written to {file_name}")




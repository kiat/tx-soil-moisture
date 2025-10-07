import pandas as pd
from datetime import datetime
from xarray import open_mfdataset
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def convert_date_to_day_of_year(date_str):
    # Parse the input date string into a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # Format the year and the day of the year into the desired format
    # %j in strftime gives the day of the year [001,366]
    converted_date = date_obj.strftime("%Y_%j")

    return converted_date

def convert_day_of_year_to_date(year_day_str):
    # Parse the input string into a datetime object
    # The format '%Y_%j' parses 'year_day of the year'
    date_obj = datetime.strptime(year_day_str, "%Y_%j")

    # Format the datetime object into a standard date string 'YYYY-MM-DD'
    standard_date = date_obj.strftime("%Y-%m-%d")

    return standard_date

def idw_interpolation(data, lat, lon, n_points=8):
    """
    Interpolates soil moisture for a given latitude and longitude using Inverse Distance Weighting (IDW).

    :param data: DataFrame containing the soil moisture data.
    :param lat: Latitude for which soil moisture is to be interpolated.
    :param lon: Longitude for which soil moisture is to be interpolated.
    :param n_points: Number of nearest points to consider for interpolation.
    :return: Interpolated soil moisture value.
    """
    # Build a KDTree for efficient nearest neighbor search
    tree = cKDTree(data[['Latitude', 'Longitude']])

    # Find the indices of the n_points nearest neighbors
    distances, indices = tree.query([lat, lon], k=n_points)

    # Avoid division by zero in case the exact point is in the dataset
    distances[distances == 0] = np.finfo(float).eps

    # Calculate weights based on inverse distance
    weights = 1 / distances

    print("Data points used for interpolation:")
    print(data.iloc[indices])
    # Calculate weighted average of soil moisture
    weighted_moisture = np.sum(weights * data.iloc[indices]['Soil_Moisture']) / np.sum(weights)

    return float(weighted_moisture)
def read_netcdf_file(dataset):
    latitudes = dataset['latitude'].values
    longitudes = dataset['longitude'].values
    global_sm = dataset['SM_daily'].values[0]
    latitudes_flat = latitudes.flatten()
    longitudes_flat = longitudes.flatten()
    soil_moisture_flat = global_sm.flatten()

    # Create a DataFrame
    df = pd.DataFrame({
        'Latitude': latitudes_flat,
        'Longitude': longitudes_flat,
        'Soil_Moisture': soil_moisture_flat
    })
    df = df.dropna(subset=['Soil_Moisture'])
    return df

csv_user_address = "/Users/jamisenma/tx-soil-moisture/satellite/satellite_data_csv/"
netcdf_user_address = "/Users/jamisenma/tx-soil-moisture/complete_satellite_data/ucar_cu_cygnss_sm_v1_"

csv_filenames = ["Station1_Satellite.csv", "Station2_Satellite.csv","Station3_Satellite.csv","Station4_Satellite.csv",
                 "Station5_Satellite.csv", "Station6_Satellite.csv",]
stations_data = [
    {'Latitude': 30.3989, 'Longitude': -98.6105, 'Station_ID': '5'},
    {'Latitude': 30.4193, 'Longitude': -98.8046, 'Station_ID': '1'},
    {'Latitude': 30.4421, 'Longitude': -98.8427, 'Station_ID': '6'},
    {'Latitude': 30.4600, 'Longitude': -98.9407, 'Station_ID': '4'},
    {'Latitude': 30.2454, 'Longitude': -98.7059, 'Station_ID': '2'},
    {'Latitude': 30.2758, 'Longitude': -98.7242, 'Station_ID': '3'}
]

null_dates_dict = {}
for i in range(0, len(csv_filenames)):
    dataframe = pd.read_csv(csv_user_address + csv_filenames[i])
    for index, row in dataframe.iterrows():
        if pd.isnull(row['soil_moisture']):
            date = convert_date_to_day_of_year(row['Date'])
            if date in null_dates_dict:
                null_dates_dict[date].append(i)
            else:
                null_dates_dict[date] = [i]
interpolation_dataframe_list = []

for i in range(0,6):
    interpolation_dataframe_list.append(pd.DataFrame())



for date in null_dates_dict:
    file_path = netcdf_user_address + date + ".dap.nc4"
    print(file_path)
    if file_path == "/Users/jamisenma/tx-soil-moisture/complete_satellite_data/ucar_cu_cygnss_sm_v1_2019_280.dap.nc4":
        continue
    try:
        dataset_file = open_mfdataset(file_path, combine="by_coords", parallel=True)
        dataset = read_netcdf_file(dataset_file)
        converted_date = convert_day_of_year_to_date(date)

        for station_number in null_dates_dict[date]:
            lat = stations_data[station_number]['Latitude']
            lon = stations_data[station_number]['Longitude']

            estimated_moisture = idw_interpolation(dataset, lat, lon)
            print(converted_date, estimated_moisture, "Station Number:", station_number)
            cur_dataframe = interpolation_dataframe_list[station_number]

            # Create a new DataFrame from the new_row dictionary
            new_row_df = pd.DataFrame([{'Date': converted_date, 'soil_moisture': estimated_moisture, 'distance': 0}])

            # Concatenate the new_row_df DataFrame with cur_dataframe
            interpolation_dataframe_list[station_number] = pd.concat([cur_dataframe, new_row_df], ignore_index=True)
    except OSError:
        print("file not found")

for i in range(0,len(csv_filenames)):
    print("Station Number:", i)
    original_dataframe = pd.read_csv(csv_user_address + csv_filenames[i])
    interpolation_dataframe = interpolation_dataframe_list[i]

    original_dataframe['Date'] = pd.to_datetime(original_dataframe['Date'])
    interpolation_dataframe['Date'] = pd.to_datetime(interpolation_dataframe['Date'])

    # Merge the DataFrames on the 'Date' column, using a left join to keep all dates from df1
    merged_df = pd.merge(original_dataframe, interpolation_dataframe, on='Date', how='left', suffixes=('', '_from_df2'))

    # Drop the 'distance_from_df2' column as it's not needed (assuming all distances in df2 are 0 and not useful)
    merged_df.drop(columns=['distance_from_df2'], inplace=True)

    # If you want to update soil_moisture in df1 with values from df2 where available
    merged_df['soil_moisture'] = merged_df['soil_moisture'].fillna(merged_df['soil_moisture_from_df2'])

    # Drop the temporary 'soil_moisture_from_df2' column
    merged_df.drop(columns=['soil_moisture_from_df2'], inplace=True)
    merged_df_filename = "/Users/jamisenma/tx-soil-moisture/satellite/satellite_data_csv/" + "Filled_Station" + str(i+1) + "_Satellite.csv"
    merged_df.to_csv(merged_df_filename, index=False)
    print(merged_df)

print(interpolation_dataframe_list)




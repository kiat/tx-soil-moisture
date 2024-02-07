import netCDF4 as nc
import pandas
import pandas as pd
import xarray
from xarray import open_mfdataset
import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from pykrige.ok import OrdinaryKriging

file_path = '/Users/jamisenma/tx-soil-moisture/complete_satellite_data/ucar_cu_cygnss_sm_v1_2020_077.dap.nc4'  # Replace with your NetCDF file path
dataset = open_mfdataset(file_path, combine="by_coords", parallel=True)


def find_closest_station(lat, lon, stations):
    distances = stations.apply(lambda station: haversine(lat, lon, station['Latitude'], station['Longitude']),
                               axis=1)
    min_index = distances.idxmin()
    return stations.iloc[min_index]['Station_ID'], distances[min_index]


def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Difference in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Earth's radius in kilometers (can be adjusted for different units)
    R = 6371.0
    distance = R * c

    return distance

#find 6 closest coordinates and make dataframe, only 1 datapoint
def make_dataframe():
    stations_data = [
        {'Latitude': 30.3989, 'Longitude': -98.6105, 'Station_ID': '5'},
        {'Latitude': 30.4193, 'Longitude': -98.8046, 'Station_ID': '1'},
        {'Latitude': 30.4421, 'Longitude': -98.8427, 'Station_ID': '6'},
        {'Latitude': 30.4600, 'Longitude': -98.9407, 'Station_ID': '4'},
        {'Latitude': 30.2454, 'Longitude': -98.7059, 'Station_ID': '2'},
        {'Latitude': 30.2758, 'Longitude': -98.7242, 'Station_ID': '3'}
    ]
    stations = pd.DataFrame(stations_data)

    # Read the variables
    # time = dataset['timeintervals'].values
    # print(time)
    # print(time.shape)
    latitudes = dataset['latitude'].values
    print(latitudes)
    print(latitudes.shape)



    longitudes = dataset['longitude'].values
    print(longitudes)
    print(longitudes.shape)
    centre_lat = 30.311826
    centre_lon = -98.775934
    global_sm = dataset['SM_daily'].values[0]
    print(global_sm)
    print(global_sm.shape)
    l3_lat = 228
    l3_lon = 107

    dataframe = pandas.DataFrame()

    latitudes_flat = latitudes.flatten()
    longitudes_flat = longitudes.flatten()
    soil_moisture_flat = global_sm.flatten()

    # Create a DataFrame
    df = pd.DataFrame({
        'Latitude': latitudes_flat,
        'Longitude': longitudes_flat,
        'Soil_Moisture': soil_moisture_flat
    })
    return df


def kriging(df):
    print("here")
    # Separate known and unknown moisture points
    known_points = df.dropna(subset=['Soil_Moisture'])
    missing_points = df[df['Soil_Moisture'].isna()]
    print("here")
    # Perform Ordinary Kriging
    ok = OrdinaryKriging(
        known_points['Longitude'].values,
        known_points['Latitude'].values,
        known_points['Soil_Moisture'].values,
        variogram_model='linear',  # You can experiment with different models
        verbose=False,
        enable_plotting=False
    )

    # Predict missing values
    z, ss = ok.execute(
        'points',
        missing_points['Longitude'].values,
        missing_points['Latitude'].values
    )

    # Fill in the missing soil moisture values
    missing_points['Soil_Moisture'] = z.data

    # Combine the known points with the newly interpolated points
    complete_df = pd.concat([known_points, missing_points], ignore_index=True)

    # Sort by Latitude and Longitude if desired
    complete_df = complete_df.sort_values(by=['Latitude', 'Longitude']).reset_index(drop=True)

    return complete_df
if __name__ == "__main__":
    dataframe = make_dataframe()
    print(kriging(dataframe))

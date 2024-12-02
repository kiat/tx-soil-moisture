# Merged Meteorological and Satellite Data

This folder contains comprehensive merged datasets combining ground-based meteorological measurements with both AMSR and SMAP satellite soil moisture observations for 6 stations in Texas.

## File Format

Each file contains daily measurements with the following columns:

1. **Date**: Date of observation (YYYY-MM-DD format)
2. **Ppt**: Daily total precipitation (mm)
3. **Srad**: Daily total solar radiation (W/m²)
4. **Tair**: Average daily air temperature (°C)
5. **RH**: Average daily relative humidity (%)
6. **Windspeed**: Average daily wind speed (m/s)
7. **Winddirection**: Average daily wind direction (degrees)
8. **Sat_SM_AMSR**: AMSR satellite soil moisture measurement (m³/m³)
9. **distance_AMSR**: Distance to AMSR measurement location (km)
10. **Sat_SM_SMAP**: SMAP satellite soil moisture measurement (m³/m³)
11. **distance_SMAP**: Distance to SMAP measurement location (km)

## Data Coverage

- Temporal coverage: 2017-2021
- Temporal resolution: Daily measurements
- AMSR data: Available throughout the period
- SMAP data: Available for later portions of the dataset
- Missing values: Indicated by empty fields

## Station Locations

The dataset includes data for 6 stations with the following GPS coordinates:

1. Latitude: 30.3989, Longitude: -98.6105
2. Latitude: 30.4193, Longitude: -98.8046
3. Latitude: 30.4421, Longitude: -98.8427
4. Latitude: 30.4600, Longitude: -98.9407
5. Latitude: 30.2454, Longitude: -98.7059
6. Latitude: 30.2758, Longitude: -98.7242

## Variable Descriptions

- **Precipitation**: Daily total rainfall in millimeters (mm)
- **Solar Radiation**: Total daily solar radiation in watts per square meter (W/m²)
- **Air Temperature**: Average daily air temperature in degrees Celsius (°C)
- **Relative Humidity**: The amount of water vapor in the air expressed as a percentage
- **Wind Speed**: Average daily wind speed in meters per second (m/s)
- **Wind Direction**: Measure

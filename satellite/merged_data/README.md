# Merged Station Data

This folder contains merged data files that combine ground-based meteorological measurements with satellite soil moisture observations for 6 stations in Texas. Each file corresponds to a different ground station location.

## Data Format

Each file contains daily measurements with the following columns:

1. **Date**: Date of observation (YYYY-MM-DD format)
2. **Sat_SM**: Satellite soil moisture measurement (m³/m³)
3. **Ppt**: Daily total precipitation (mm)
4. **Tair**: Average daily air temperature (°C)
5. **RH**: Average daily relative humidity (%)
6. **Windspeed**: Average daily wind speed (m/s)
7. **Winddirection**: Average daily wind direction (degrees)
8. **Srad**: Daily total solar radiation (W/m²)
9. **Latitude**: Station latitude (constant for each station)
10. **Longitude**: Station longitude (constant for each station)

## Data Coverage

- Temporal coverage: 2017-2020
- Temporal resolution: Daily measurements
- Spatial coverage: 6 stations in Texas

## Station Locations

The dataset includes data for 6 stations with the following GPS coordinates:

1. Latitude: 30.3989, Longitude: -98.6105
2. Latitude: 30.4193, Longitude: -98.8046
3. Latitude: 30.4421, Longitude: -98.8427
4. Latitude: 30.4600, Longitude: -98.9407
5. Latitude: 30.2454, Longitude: -98.7059
6. Latitude: 30.2758, Longitude: -98.7242

## Variable Descriptions

- **Sat_SM**: Satellite-based soil moisture measurement (m³/m³), typically ranging from 0.0 to 0.4
- **Precipitation (Ppt)**: Daily total rainfall in millimeters (mm)
- **Air Temperature (Tair)**: Average daily air temperature in degrees Celsius (°C)
- **Relative Humidity (RH)**: The amount of water vapor in the air expressed as a percentage
- **Wind Speed**: Average daily wind speed in meters per second (m/s)
- **Wind Direction**: Measured in degrees, where 0° (or 360°) represents North, 90° East, 180° South, and 270° West
- **Solar Radiation (Srad)**: Total daily solar radiation in watts per square meter (W/m²)

## Data Processing Notes

1. All meteorological variables are derived from ground-based measurements
2. Satellite soil moisture data is from merged satellite observations
3. Daily values are calculated as follows:
   - Precipitation and Solar Radiation are daily totals
   - All other variables are daily averages

## File Naming Convention

Files follow the naming pattern: `StationX_Merged.csv`, where X is the station number (1-6).

## Data Applications

This merged dataset is suitable for:
- Validation of satellite soil moisture products
- Studies of soil moisture-climate relationships
- Agricultural and hydrological modeling
- Climate monitoring and analysis

## Related Datasets

The raw data can be found in:
1. AMSR satellite data directory
2. Ground-based meteorological measurements
3. Station metadata files

# AMSR Merged Dataset

This folder contains merged data files combining AMSR satellite soil moisture measurements with ground-based meteorological observations for 6 stations in Texas. Each file corresponds to a different ground station location.

## File Format

Each file contains daily measurements with the following columns:

1. **Date**: Date of observation (YYYY-MM-DD format)
2. **soil_moisture**: AMSR satellite soil moisture measurement (m³/m³)
3. **Precipitation**: Daily precipitation total (mm)
4. **Temperature**: Average daily temperature (°C)
5. **Relative Humidity**: Average daily relative humidity (%)
6. **Wind Speed**: Average daily wind speed (m/s)
7. **Solar Radiation**: Daily total solar radiation (W/m²)
8. **Wind Direction**: Average daily wind direction (degrees)
9. **Latitude**: Station latitude (constant for each station)
10. **Longitude**: Station longitude (constant for each station)

## Data Coverage

- Temporal coverage: 2017-2020
- Temporal resolution: Daily measurements
- Spatial coverage: 6 stations in Texas
- Missing data: Some dates may be missing due to satellite orbital patterns or data quality issues

## Station Locations

The dataset includes data for 6 stations with the following GPS coordinates:

1. Latitude: 30.3989, Longitude: -98.6105
2. Latitude: 30.4193, Longitude: -98.8046
3. Latitude: 30.4421, Longitude: -98.8427
4. Latitude: 30.4600, Longitude: -98.9407
5. Latitude: 30.2454, Longitude: -98.7059
6. Latitude: 30.2758, Longitude: -98.7242

## Variable Descriptions

- **Soil Moisture**: Dimensionless property representing the volume of water in a given volume of soil (m³/m³). Values typically range from 0.0 to 0.4.
- **Precipitation**: Daily total rainfall in millimeters (mm)
- **Temperature**: Average daily air temperature in degrees Celsius (°C)
- **Relative Humidity**: The amount of water vapor in the air expressed as a percentage
- **Wind Speed**: Average daily wind speed in meters per second (m/s)
- **Solar Radiation**: Total daily solar radiation in watts per square meter (W/m²)
- **Wind Direction**: Measured in degrees, where 0° (or 360°) represents North, 90° East, 180° South, and 270° West

## Data Processing

This dataset merges:
1. AMSR satellite soil moisture observations
2. Ground-based meteorological measurements
3. Station location information

The merged dataset provides a comprehensive view of soil moisture conditions and related meteorological variables, making it suitable for:
- Validation of satellite measurements
- Studies of soil moisture dynamics
- Analysis of meteorological impacts on soil moisture
- Development and testing of prediction models

## File Naming Convention

Files follow the naming pattern: `AMSR_station_X_merged.csv`, where X is the station number (1-6).

## Related Datasets

For satellite-only measurements, refer to the files in the AMSR_data_csv directory.

# Data Summary

This dataset contains soil moisture and meteorological data from 6 different stations located in Texas. The data is processed and merged from multiple sources, including ground-based sensors and satellite observations.

## Data Sources

1. Ground-based soil moisture and temperature measurements (SM_1.dat to SM_6.dat)
2. Ground-based meteorological measurements (MET_1.dat to MET_6.dat)
3. AMSR satellite soil moisture data
4. SMAP satellite soil moisture data

## Data Processing

The original hourly data has been aggregated to daily values:

- Precipitation (Ppt) and Solar Radiation (Srad) are summed for each day.
- All other variables are averaged for each day.

## Variables

1. **Date**: Date of observation
2. **Ppt**: Daily total precipitation (mm)
3. **Srad**: Daily total solar radiation (W/m²)
4. **SWC_5, SWC_10, SWC_20, SWC_50**: Average daily soil water content at 5, 10, 20, and 50 cm depths (m³/m³)
5. **T_5, T_10, T_20, T_50**: Average daily soil temperature at 5, 10, 20, and 50 cm depths (°C)
6. **Tair**: Average daily air temperature (°C)
7. **RH**: Average daily relative humidity (%)
8. **Windspeed**: Average daily wind speed (m/s)
9. **Winddirection**: Average daily wind direction (degrees)
10. **Sat_SM_AMSR**: AMSR satellite soil moisture measurement (if available)
11. **distance_AMSR**: Distance to AMSR measurement location
12. **Sat_SM_SMAP**: SMAP satellite soil moisture measurement (if available)
13. **distance_SMAP**: Distance to SMAP measurement location

## Variable Descriptions

- **Soil Moisture (SWC)**: Dimensionless property representing the volume of water in a given volume of soil (m³/m³). Values typically range from 0.0 to 0.6, but can theoretically be between 0.0 and 1.0.
- **Relative Humidity (RH)**: The amount of water vapor in the air expressed as a percentage of the amount needed for saturation at the same temperature.
- **Solar Radiation (Srad)**: The power per unit area received from the Sun in the form of electromagnetic radiation.
- **Wind Direction**: Measured in degrees, where 0° (or 360°) represents North, 90° East, 180° South, and 270° West.

## Data Filtering

The merged dataset only includes dates that exist in both the ground-based measurements AND either the AMSR or SMAP satellite data. This ensures that each data point has both ground and satellite measurements for comparison and analysis.

## Station Locations

The dataset includes data from 6 stations with the following GPS coordinates:

1. Latitude: 30.3989, Longitude: -98.6105
2. Latitude: 30.4193, Longitude: -98.8046
3. Latitude: 30.4421, Longitude: -98.8427
4. Latitude: 30.4600, Longitude: -98.9407
5. Latitude: 30.2454, Longitude: -98.7059
6. Latitude: 30.2758, Longitude: -98.7242

This merged dataset provides a comprehensive view of soil moisture conditions, combining high-resolution ground measurements with broader satellite observations, making it suitable for various hydrological and meteorological analyses.
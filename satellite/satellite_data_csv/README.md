# Satellite Soil Moisture Data

This folder contains satellite-based soil moisture measurements for 6 stations in Texas. Each file contains daily soil moisture observations from satellite measurements.

## File Format

Each file contains daily measurements with the following columns:

1. **Date**: Date of observation (YYYY-MM-DD format)
2. **soil_moisture**: Satellite soil moisture measurement (m³/m³)
3. **distance**: Distance between ground station and satellite measurement location (km)

## Data Characteristics

- Temporal coverage: 2017-2022
- Temporal resolution: Daily measurements
- Missing data: Indicated by empty fields
- Soil moisture values typically range from 0.09 to 0.25 m³/m³
- Consistent measurement distance of approximately 12.27 km for each station

## Station Locations

The dataset includes data for 6 stations with the following GPS coordinates:

1. Latitude: 30.3989, Longitude: -98.6105
2. Latitude: 30.4193, Longitude: -98.8046
3. Latitude: 30.4421, Longitude: -98.8427
4. Latitude: 30.4600, Longitude: -98.9407
5. Latitude: 30.2454, Longitude: -98.7059
6. Latitude: 30.2758, Longitude: -98.7242

## Data Quality Notes

1. Gaps in the data occur due to:
   - Satellite orbital patterns
   - Quality control filtering
   - Atmospheric interference
   - Cloud cover

2. Distance values remain constant for each station as they represent the fixed distance between the ground station and the satellite measurement footprint center

## File Naming Convention

Files follow the naming pattern: `StationX_Satellite.csv`, where X is the station number (1-6).

## Related Datasets

For comprehensive datasets including meteorological measurements, refer to:
1. Merged meteorological and satellite data files
2. Ground-based measurements
3. AMSR-SMAP combined datasets
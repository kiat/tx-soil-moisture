# AMSR Satellite Data

This folder contains soil moisture data from the AMSR (Advanced Microwave Scanning Radiometer) satellite for 6 different stations located in Texas. Each file corresponds to a different ground station location.

## Data Format

Each file contains daily AMSR satellite measurements with the following columns:

1. **Date**: Date of observation (YYYY-MM-DD format)
2. **soil_moisture**: AMSR satellite soil moisture measurement (m³/m³)
3. **distance**: Distance between the satellite measurement location and the ground station (km)

## Data Coverage

- Temporal coverage: 2017-2022
- Temporal resolution: Daily measurements
- Missing data: Some dates may be missing due to satellite orbital patterns or data quality issues

## Station Locations

The dataset includes data for 6 stations with the following GPS coordinates:

1. Latitude: 30.3989, Longitude: -98.6105
2. Latitude: 30.4193, Longitude: -98.8046
3. Latitude: 30.4421, Longitude: -98.8427
4. Latitude: 30.4600, Longitude: -98.9407
5. Latitude: 30.2454, Longitude: -98.7059
6. Latitude: 30.2758, Longitude: -98.7242

## Data Characteristics

### Soil Moisture
- Unit: m³/m³ (volumetric water content)
- Typical range: 0.0 to 0.4
- Resolution: ~0.001 m³/m³

### Distance
- Unit: kilometers (km)
- Range: Typically 9-700 km
- Smaller distances generally indicate more reliable measurements
- Large distances (>100 km) may indicate non-optimal satellite coverage for that day

## Data Quality Notes

1. Measurements with very large distances (>300 km) should be used with caution as they may not accurately represent conditions at the ground station
2. Soil moisture values near 0 or above 0.5 may indicate measurement artifacts
3. Each file follows the naming convention: AMSR_station_X_data.csv, where X is the station number (1-6)

## Usage Notes

This data is typically used in conjunction with ground-based measurements for:
- Validation of satellite soil moisture products
- Development of soil moisture retrieval algorithms
- Studies of soil moisture patterns and variability
- Hydrological modeling and drought monitoring

## Related Datasets

This data can be combined with:
1. Ground-based soil moisture measurements
2. SMAP satellite observations
3. Local meteorological data
4. Other environmental variables in the all_merged_data folder

For a complete merged dataset containing all variables, refer to the files in the all_merged_data directory.

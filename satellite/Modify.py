import pandas as pd
from datetime import datetime
import os

def process_station(station_number):
    print(f"\nProcessing Station {station_number}")
    
    # Read the simulated data file
    df_sim = pd.read_csv(f'datasets/Merged-Cleaned-Simulated-Data/Station{station_number}-Simulated-cleaned-merged-data.csv', parse_dates=['Date'])

    # Read AMSR and SMAP data
    amsr_df = pd.read_csv(f'satellite/AMSR_data_csv/AMSR_station_{station_number}_data.csv', parse_dates=['Date'])
    smap_df = pd.read_csv(f'satellite/satellite_data_csv/Station{station_number}_Satellite.csv', parse_dates=['Date'])

    # Convert all dates to date only (no time component)
    df_sim['Date'] = df_sim['Date'].dt.date
    amsr_df['Date'] = pd.to_datetime(amsr_df['Date']).dt.date
    smap_df['Date'] = pd.to_datetime(smap_df['Date']).dt.date

    # Get the set of dates that exist in simulated data AND (AMSR OR SMAP)
    sim_dates = set(df_sim['Date'])
    satellite_dates = set(amsr_df['Date']) | set(smap_df['Date'])
    valid_dates = sim_dates & satellite_dates

    # Filter the simulated data to include only valid dates
    df_sim = df_sim[df_sim['Date'].isin(valid_dates)]

    # Group by date and calculate daily statistics
    daily_stats = df_sim.groupby('Date').agg({
        'Ppt': 'sum',
        'Srad': 'sum',
        'SWC_5': 'mean',
        'SWC_10': 'mean',
        'SWC_20': 'mean',
        'SWC_50': 'mean',
        'T_5': 'mean',
        'T_10': 'mean',
        'T_20': 'mean',
        'T_50': 'mean',
        'Tair': 'mean',
        'RH': 'mean',
        'Windspeed': 'mean',
        'Winddirection': 'mean'
    }).reset_index()

    # Merge with AMSR and SMAP data
    merged_df = pd.merge(daily_stats, amsr_df[['Date', 'soil_moisture', 'distance']], on='Date', how='left')
    merged_df = pd.merge(merged_df, smap_df[['Date', 'soil_moisture', 'distance']], on='Date', how='left', suffixes=('_AMSR', '_SMAP'))

    # Rename columns
    merged_df = merged_df.rename(columns={
        'soil_moisture_AMSR': 'Sat_SM_AMSR',
        'soil_moisture_SMAP': 'Sat_SM_SMAP',
        'distance_AMSR': 'distance_AMSR',
        'distance_SMAP': 'distance_SMAP'
    })

    # Drop Latitude and Longitude columns if they exist
    merged_df = merged_df.drop(columns=['Latitude', 'Longitude'], errors='ignore')

    # Sort the merged dataframe by date
    merged_df = merged_df.sort_values('Date')

    # Save the modified dataframe back to the original CSV file
    merged_df.to_csv(f'satellite/all_merged_data/Station{station_number}_AMSR_SMAP_Merged.csv', index=False, date_format='%Y-%m-%d')

    print(f"File for Station {station_number} has been successfully modified and saved.")

# Process all 6 stations
for station in range(1, 7):
    process_station(station)

print("\nAll stations have been processed.")

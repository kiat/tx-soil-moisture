import pandas as pd
from datetime import datetime
import os

def process_station(station_number):
    print(f"\nProcessing Station {station_number}")
    
    # Read the first CSV file
    df1 = pd.read_csv(f'satellite/all_merged_data/Station{station_number}_AMSR_SMAP_Merged.csv', parse_dates=['Date'])
    print(f"df1 shape: {df1.shape}")
    print(f"df1 date range: {df1['Date'].min()} to {df1['Date'].max()}")

    # Read the second CSV file
    df2 = pd.read_csv(f'datasets/Merged-Cleaned-Simulated-Data/Station{station_number}-Simulated-cleaned-merged-data.csv', parse_dates=['Date'])
    print(f"df2 shape: {df2.shape}")
    print(f"df2 date range: {df2['Date'].min()} to {df2['Date'].max()}")

    # Filter df2 for only noon data
    df2_filtered = df2[df2['Date'].dt.hour == 12].copy()
    df2_filtered['Date'] = df2_filtered['Date'].dt.date
    print(f"df2_filtered shape: {df2_filtered.shape}")
    print(f"df2_filtered date range: {df2_filtered['Date'].min()} to {df2_filtered['Date'].max()}")

    # Convert df1's Date to date only
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date

    # Read AMSR and SMAP data
    amsr_df = pd.read_csv(f'satellite/AMSR_data_csv/AMSR_station_{station_number}_data.csv', parse_dates=['Date'])
    smap_df = pd.read_csv(f'satellite/satellite_data_csv/Station{station_number}_Satellite.csv', parse_dates=['Date'])

    # Convert AMSR and SMAP Date to date only
    amsr_df['Date'] = pd.to_datetime(amsr_df['Date']).dt.date
    smap_df['Date'] = pd.to_datetime(smap_df['Date']).dt.date

    # Merge the filtered data from df2 into df1
    merged_df = pd.concat([df1, df2_filtered[['Date', 'SWC_5', 'SWC_10', 'SWC_20', 'SWC_50', 'T_5', 'T_10', 'T_20', 'T_50', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'Srad', 'Ppt']]])

    # Sort the merged dataframe by date
    merged_df = merged_df.sort_values('Date')

    # Remove duplicate dates, keeping the data from df2_filtered if available
    merged_df = merged_df.drop_duplicates(subset=['Date'], keep='last')

    # Merge with AMSR and SMAP data
    merged_df = pd.merge(merged_df, amsr_df[['Date', 'distance']], on='Date', how='left', suffixes=('', '_AMSR'))
    merged_df = pd.merge(merged_df, smap_df[['Date', 'distance']], on='Date', how='left', suffixes=('', '_SMAP'))

    # Rename columns
    merged_df = merged_df.rename(columns={'distance': 'distance_AMSR', 'distance_SMAP': 'distance_SMAP'})

    print(f"Final merged_df shape: {merged_df.shape}")
    print(f"Final merged_df date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")

    # Save the modified dataframe back to the original CSV file
    merged_df.to_csv(f'satellite/all_merged_data/Station{station_number}_AMSR_SMAP_Merged.csv', index=False, date_format='%Y-%m-%d')

    print(f"File for Station {station_number} has been successfully modified and saved.")

# Process all 6 stations
for station in range(1, 7):
    process_station(station)

print("\nAll stations have been processed.")

import csv
import os
from datetime import datetime, time
import pandas as pd

def parse_date(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d')

def merge_datasets(revised_final_file, smap_file, amsr_file, output_file):
    # Read the revised final data
    revised_final_df = pd.read_csv(revised_final_file, parse_dates=['Unnamed: 0'])
    revised_final_df.set_index('Unnamed: 0', inplace=True)
    revised_final_df.index.name = 'Date'

    # Select only 12 PM data from each day
    revised_final_df = revised_final_df.at_time('12:00')

    # Read the SMAP file
    smap_df = pd.read_csv(smap_file, parse_dates=['Date'])
    smap_df.set_index('Date', inplace=True)
    smap_df.rename(columns={'soil_moisture': 'Sat_SM_SMAP'}, inplace=True)

    # Read the AMSR file
    amsr_df = pd.read_csv(amsr_file, parse_dates=['Date'])
    amsr_df.set_index('Date', inplace=True)
    amsr_df.rename(columns={'soil_moisture': 'Sat_SM_AMSR'}, inplace=True)

    # Merge all dataframes
    merged_df = revised_final_df.join([smap_df['Sat_SM_SMAP'], amsr_df['Sat_SM_AMSR']], how='outer')

    # Combine midnight and noon data
    merged_df.reset_index(inplace=True)
    merged_df['Date'] = pd.to_datetime(merged_df['Date'].dt.date)  # Remove time component
    merged_df = merged_df.groupby('Date').first().reset_index()  # Keep first non-null value for each day

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the merged data to the output file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged data written to {output_file}")

# Process all 6 stations
for i in range(1, 7):
    revised_final_file = f'datasets/Revised_Final_Data/Station{i}_Revised_Final_Data.csv'
    smap_file = f'satellite/satellite_data_csv/Station{i}_Satellite.csv'
    amsr_file = f'satellite/AMSR_data_csv/AMSR_station_{i}_data.csv'
    output_file = f'satellite/all_merged_data/Station{i}_AMSR_SMAP_Merged.csv'
    
    merge_datasets(revised_final_file, smap_file, amsr_file, output_file)

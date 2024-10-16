import csv
import os
from datetime import datetime

def parse_date(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d')

def merge_datasets(merged_file, smap_file, amsr_file, output_file):
    data = {}

    # Read the merged SMAP file
    with open(merged_file, 'r') as f:
        reader = csv.reader(f)
        headers_merged = next(reader)  # Save headers
        sat_sm_index = headers_merged.index('Sat_SM')
        headers_merged = headers_merged[:sat_sm_index] + headers_merged[sat_sm_index+1:]  # Remove Sat_SM from headers
        for row in reader:
            if len(row) > 0:
                date = parse_date(row[0])
                # Remove Sat_SM column from data
                data[date] = row[1:sat_sm_index+1] + row[sat_sm_index+2:]

    # Read the SMAP file to get SMAP soil moisture
    smap_sm = {}
    with open(smap_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) > 0:
                date = parse_date(row[0])
                smap_sm[date] = row[1]  # SMAP soil moisture is in the second column

    # Read the AMSR file to get AMSR soil moisture
    amsr_sm = {}
    with open(amsr_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) > 0:
                date = parse_date(row[0])
                amsr_sm[date] = row[1]  # AMSR soil moisture is in the second column

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the merged data to the output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Add new headers for SMAP and AMSR soil moisture
        combined_headers = ['Date'] + headers_merged + ['Sat_SM_SMAP', 'Sat_SM_AMSR']
        writer.writerow(combined_headers)
        
        for date in sorted(data.keys()):
            row = data[date]
            smap_sm_value = smap_sm.get(date, '')
            amsr_sm_value = amsr_sm.get(date, '')
            writer.writerow([date.strftime('%Y-%m-%d')] + row + [smap_sm_value, amsr_sm_value])

    print(f"Merged data written to {output_file}")

# Process all 6 stations
for i in range(1, 7):
    merged_file = f'satellite/merged_data/Station{i}_Merged.csv'
    smap_file = f'satellite/satellite_data_csv/Station{i}_Satellite.csv'
    amsr_file = f'satellite/AMSR_data_csv/AMSR_station_{i}_data.csv'
    output_file = f'satellite/all_merged_data/Station{i}_AMSR_SMAP_Merged.csv'
    
    merge_datasets(merged_file, smap_file, amsr_file, output_file)

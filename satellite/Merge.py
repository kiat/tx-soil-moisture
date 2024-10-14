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
        for row in reader:
            if len(row) > 0:
                date = parse_date(row[0])
                data[date] = row

    # Read the SMAP file to get distance
    smap_distances = {}
    with open(smap_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) > 0:
                date = parse_date(row[0])
                smap_distances[date] = row[2]  # distance is in the third column

    # Read the AMSR file to get distance
    amsr_distances = {}
    with open(amsr_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) > 0:
                date = parse_date(row[0])
                amsr_distances[date] = row[2]  # distance is in the third column

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the merged data to the output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Add new headers for SMAP and AMSR distances
        combined_headers = headers_merged + ['distance_SMAP', 'distance_AMSR']
        writer.writerow(combined_headers)
        for date in sorted(data.keys()):
            row = data[date]
            smap_distance = smap_distances.get(date, '')
            amsr_distance = amsr_distances.get(date, '')
            writer.writerow(row + [smap_distance, amsr_distance])

    print(f"Merged data written to {output_file}")

# Process all 6 stations
for i in range(1, 7):
    merged_file = f'satellite/merged_data/Station{i}_Merged.csv'
    smap_file = f'satellite/satellite_data_csv/Station{i}_Satellite.csv'
    amsr_file = f'satellite/AMSR_data_csv/AMSR_station_{i}_data.csv'
    output_file = f'satellite/all_merged_data/Station{i}_AMSR_SMAP_Merged.csv'
    
    merge_datasets(merged_file, smap_file, amsr_file, output_file)

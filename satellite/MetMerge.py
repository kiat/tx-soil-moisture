import pandas as pd
import os

def process_station_data(station_number):
    # Input and output paths
    input_file = f'satellite/all_merged_data/Station{station_number}_AMSR_SMAP_Merged.csv'
    output_dir = 'satellite/met_merged_satelite_data'
    output_file = f'{output_dir}/Station{station_number}_Met_AMSR_SMAP.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # List of soil-related columns to remove
        soil_columns = [
            'SWC_5', 'SWC_10', 'SWC_20', 'SWC_50',
            'T_5', 'T_10', 'T_20', 'T_50'
        ]
        
        # Remove soil-related columns
        df = df.drop(columns=soil_columns)
        
        # Save to new location
        df.to_csv(output_file, index=False)
        print(f"Successfully processed Station {station_number}")
        
    except Exception as e:
        print(f"Error processing Station {station_number}: {str(e)}")

def main():
    print("Starting to process station data...")
    
    # Process stations 1 through 6
    for station in range(1, 7):
        process_station_data(station)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()

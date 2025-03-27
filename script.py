import tensorflow as tf 
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import glob
import re
import seaborn as sns
import argparse

def clean_data(sm_df, met_df):

    # Ensure index is in datetime format
    sm_df.index = pd.to_datetime(sm_df.index, errors='coerce')
    met_df.index = pd.to_datetime(met_df.index, errors='coerce')

    # Drop rows where datetime parsing failed (invalid dates)
    sm_df = sm_df.dropna()
    met_df = met_df.dropna()

    # Strip whitespace from column names
    sm_df.columns = sm_df.columns.str.replace(' ', '')
    met_df.columns = met_df.columns.str.replace(' ', '')

    # Rename "Ppt" columns before merging
    sm_df = sm_df.rename(columns={"Ppt": "Ppt_sm"})
    met_df = met_df.rename(columns={"Ppt": "Ppt_met"})

    # Check to see if all hours of every day are counted
    full_date_range = pd.date_range(start=sm_df.index.min(), end=sm_df.index.max(), freq='H')

    # Reindex to include all dates, filling missing ones with NaN
    sm_df = sm_df.reindex(full_date_range)
    met_df = met_df.reindex(full_date_range)

    # Find missing dates
    missing_dates = sm_df[sm_df.isnull().all(axis=1)].index

    # Display missing dates
    if not missing_dates.empty:
        print("Missing Dates (now filled with NaN):", missing_dates.tolist())
    else:
        print("No missing dates found.")

    # Convert numeric columns to float, handling extra spaces and commas
    def clean_numeric(value):
        try:
            return float(str(value).replace(',', '').strip())
        except ValueError:
            return np.nan  # Replace invalid values with NaN

    sm_df = sm_df.applymap(clean_numeric)
    met_df = met_df.applymap(clean_numeric)

    # Drop 'Flag' column if present
    if 'Flag' in sm_df.columns:
        sm_df = sm_df.drop(columns=['Flag'])

    return sm_df, met_df

def calculate_missing_values(sm_df, met_df, output_file="list_of_missing_values.csv"):
    missing_values_list = []

    for col in sm_df.columns.tolist() + met_df.columns.tolist():
        missing_rows = sm_df[sm_df[col].isna()] if col in sm_df.columns else met_df[met_df[col].isna()]
        
        for timestamp in missing_rows.index:
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")  # Format timestamp
            missing_values_list.append([formatted_timestamp, col])

    missing_values_df = pd.DataFrame(missing_values_list, columns=["Timestamp", "Attribute"])

    # Save as CSV
    missing_values_df.to_csv(output_file, index=False)
    print(f"\nMissing values exported to {output_file}")

def calculate_missing_value_cooccurrence(sm_df, met_df, output_file="missing_values_cooccurrence.csv"):
    # Initialize an empty list to store the missing values data
    missing_values_list = []

    # Iterate through each pair of columns from both sm_df and met_df
    for col1 in sm_df.columns.tolist() + met_df.columns.tolist():
        for col2 in sm_df.columns.tolist() + met_df.columns.tolist():
            # Find the rows where both columns have missing values
            if col1 in sm_df.columns and col2 in sm_df.columns:
                missing_rows = sm_df[sm_df[col1].isna() & sm_df[col2].isna()]
            elif col1 in met_df.columns and col2 in met_df.columns:
                missing_rows = met_df[met_df[col1].isna() & met_df[col2].isna()]
            elif col1 in sm_df.columns and col2 in met_df.columns:
                missing_rows = sm_df[sm_df[col1].isna() & met_df[col2].isna()]
            elif col1 in met_df.columns and col2 in sm_df.columns:
                missing_rows = met_df[met_df[col1].isna() & sm_df[col2].isna()]

            # If there are missing values, append the timestamp and column names to the list
            for timestamp in missing_rows.index:
                missing_values_list.append([timestamp, col1, col2])  # Removed the missing count

    # Convert the list of missing values to a DataFrame
    missing_values_df = pd.DataFrame(missing_values_list, columns=["Timestamp", "Variable_1", "Variable_2"])

    # Export the missing values DataFrame to a CSV file
    missing_values_df.to_csv(output_file, index=False)
    print(f"Missing values co-occurrence saved to {output_file}")

def get_station_summary(station, sm_df, met_df):
    # metric names
    sm_metrics = ["Ppt_sm", "SWC_5", "SWC_10", "SWC_20", "SWC_50", "T_5", "T_10", "T_20", "T_50"]
    met_metrics = ["Ppt_met", "Tair", "RH", "Windspeed", "Winddirection", "Srad"]

    # --- Process SM Data ---
    n_rows_sm = sm_df.shape[0]
    sm_rows = []
    # Header row for SM Data (just a label row; you can leave the numeric columns blank)
    sm_rows.append(("SM Data", "", "", ""))
    sm_missing_total = 0
    for metric in sm_metrics:
        if metric in sm_df.columns:
            total = n_rows_sm
            missing = sm_df[metric].isna().sum()
        else:
            total = n_rows_sm
            missing = 0
        pct = round((missing / total * 100), 2) if total else 0
        sm_rows.append((metric, total, missing, pct))
        sm_missing_total += missing
    # TOTAL row for SM (aggregate over all SM metrics)
    total_possible_sm = n_rows_sm * len(sm_metrics)
    total_pct_sm = round((sm_missing_total / total_possible_sm * 100), 2) if total_possible_sm else 0
    sm_rows.append(("TOTAL", total_possible_sm, sm_missing_total, total_pct_sm))
    sm_summary = pd.DataFrame(sm_rows, columns=["Metric", "Total Values", "Missing Values", "Missing (%)"])
    sm_summary.set_index("Metric", inplace=True)

    # --- Process MET Data ---
    n_rows_met = met_df.shape[0]
    met_rows = []
    met_rows.append(("MET Data", "", "", ""))
    met_missing_total = 0
    for metric in met_metrics:
        if metric in met_df.columns:
            total = n_rows_met
            missing = met_df[metric].isna().sum()
        else:
            total = n_rows_met
            missing = 0
        pct = round((missing / total * 100), 2) if total else 0
        met_rows.append((metric, total, missing, pct))
        met_missing_total += missing
    total_possible_met = n_rows_met * len(met_metrics)
    total_pct_met = round((met_missing_total / total_possible_met * 100), 2) if total_possible_met else 0
    met_rows.append(("TOTAL", total_possible_met, met_missing_total, total_pct_met))
    met_summary = pd.DataFrame(met_rows, columns=["Metric", "Total Values", "Missing Values", "Missing (%)"])
    met_summary.set_index("Metric", inplace=True)

    # --- Combine SM and MET summaries vertically ---
    station_summary = pd.concat([sm_summary, met_summary])
    return station_summary

# Main Method Below
# Dictionary to collect summaries per station
station_summaries = {}

# Create an output folder for summaries if it doesn't exist
output_folder = "station_summaries_output"
os.makedirs(output_folder, exist_ok=True)

# Folder paths (updated)
# Automatically find the paths for soil_station folders
base_folder = "./datasets/TX-Data"
sm_folder = os.path.join(base_folder, "soil_station")
met_folder = os.path.join(base_folder, "met_station")

# Get all station numbers from filenames (assuming format: SM_101.dat, MET_101.dat)
station_numbers = sorted(set(
    int(re.search(r"SM_(\d+)\.dat", f).group(1))
    for f in os.listdir(sm_folder) if re.match(r"SM_\d+\.dat", f)
))

# Now process each station
for station in station_numbers:
    sm_path = os.path.join(sm_folder, f"SM_{station}.dat")
    met_path = os.path.join(met_folder, f"MET_{station}.dat")
    
    # Load data
    sm_df = pd.read_csv(sm_path, sep=",", parse_dates=["Date"], index_col="Date") 
    met_df = pd.read_csv(met_path, sep=",", parse_dates=["Date"], index_col="Date") 

    # Check if dataframes are empty
    if sm_df.empty:
        print(f"Warning: SM_{station}.dat is empty. Skipping this station.")
        continue
    if met_df.empty:
        print(f"Warning: MET_{station}.dat is empty. Skipping this station.")
        continue

    print(f"\nProcessing Station {station}...")

    # Data cleaning
    sm_df, met_df = clean_data(sm_df, met_df)
    
    # Define station-specific output file paths for missing values and co-occurrence
    missing_values_file = os.path.join(output_folder, f"missing_values_station_{station}.csv")
    cooccurrence_file = os.path.join(output_folder, f"missing_values_cooccurrence_station_{station}.csv")

    # Calculate missing values and save per station
    calculate_missing_values(sm_df, met_df, output_file=missing_values_file)

    # Calculate missing value co-occurrence and save per station
    calculate_missing_value_cooccurrence(sm_df, met_df, output_file=cooccurrence_file)

    # Store station summary in dictionary
    station_summaries[station] = get_station_summary(station, sm_df, met_df)

    # Save station-specific summary as CSV
    station_summary_file = os.path.join(output_folder, f"summary_station_{station}.csv")
    station_summaries[station].to_csv(station_summary_file)

    print(f"Completed processing for Station {station}. Data saved in '{output_folder}'\n")

# After processing all stations, no need for a combined summary
print(f"\n All station summaries and missing value files saved in '{output_folder}'")

# Set the style for the plots
sns.set_style("whitegrid")

# Function to loop over both directories and process files //netha added, take w grain of salt
def process_all_weather_data(sm_folder, met_folder):
    # Get all the filenames in the directories
    sm_files = [f for f in os.listdir(sm_folder) if f.endswith('.dat')]
    met_files = [f for f in os.listdir(met_folder) if f.endswith('.dat')]

    # Process each file in sm_folder and met_folder
    for sm_file in sm_files:
        sm_file_path = os.path.join(sm_folder, sm_file)
        print(f"\nProcessing SM File: {sm_file_path}")
        process_weather_data(sm_file_path, is_sm_file=True)

    for met_file in met_files:
        met_file_path = os.path.join(met_folder, met_file)
        print(f"\nProcessing MET File: {met_file_path}")
        process_weather_data(met_file_path, is_sm_file=False)


output_folder = "invalid_data_summaries"
os.makedirs(output_folder, exist_ok=True)


def process_weather_data(file_path, is_sm_file=False):
    """
    Loads, cleans, validates, and visualizes weather data from a given file.

    Parameters:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame: DataFrame containing invalid data points.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert relevant columns to numeric
    if not is_sm_file:
        numeric_cols = ['Tair', 'RH', 'Wind speed', 'Wind direction', 'Srad']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Ensure the 'Date' column is in datetime format, if it's not already
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

  
    # Define validation function
    def validate_data(df):
        mask = pd.Series(False, index=df.index)  # Initialize mask with all False values

        # Convert columns to numeric (force non-numeric values to NaN)
        if not is_sm_file:
            numeric_cols = ['Tair', 'RH', 'Wind speed', 'Wind direction', 'Srad']
        else:
            numeric_cols = ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coerce errors to NaN

        # Apply validation rules
        if not is_sm_file:
            mask |= (
                ((df['Tair'] < -50) | (df['Tair'] > 50)) |
                ((df['RH'] < 0) | (df['RH'] > 100)) |
                ((df['Wind speed'] < 0) | (df['Wind speed'] > 25)) |
                ((df['Wind direction'] < 0) | (df['Wind direction'] > 360)) |
                (df['Srad'] < 0)
            )

        # Apply SWC validation only for SM files
        if is_sm_file:
            for col in ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']:
                if col in df.columns:
                    mask |= ((df[col] < 0.0) | (df[col] > 0.6))

        return df[mask]



    # Run validation
    invalid_data = validate_data(df)
    
    if not invalid_data.empty:
        print(f"Invalid Data Points in {file_path}:")

        # Prepare data for CSV export
        invalid_rows = []
        for index, row in invalid_data.iterrows():
            if pd.notna(row['Date']):
                date_hour = row['Date'].strftime('%Y-%m-%d %H:%M')
            else:
                date_hour = 'N/A'
            
            issues = []
            if not is_sm_file:
                for col in row.index:
                    if pd.isna(row[col]):
                        issues.append(f"Missing value in {col}")
                    elif col == 'Tair' and (row[col] < -50 or row[col] > 50):
                        issues.append(f"Invalid temperature: {row[col]}°C")
                    elif col == 'RH' and (row[col] < 0 or row[col] > 100):
                        issues.append(f"Invalid humidity: {row[col]}%")
                    elif col == 'Wind speed' and (row[col] < 0 or row[col] > 25):
                        issues.append(f"Invalid wind speed: {row[col]} m/s")
                    elif col == 'Wind direction' and (row[col] < 0 or row[col] > 360):
                        issues.append(f"Invalid wind direction: {row[col]}°")
                    elif col == 'Srad' and row[col] < 0:
                        issues.append(f"Invalid solar radiation: {row[col]} W/m²")

            # Print each issue
                # print(f"Index: {index} | Date & Hour: {date_hour} | Issues: {', '.join(issues)}")

            # Store in list for CSV export
                invalid_rows.append([index, date_hour, "; ".join(issues)])

        # Save invalid data to CSV
        file_name = os.path.basename(file_path).replace('.dat', '_invalid_data.csv')
        output_file = os.path.join(output_folder, file_name)
        pd.DataFrame(invalid_rows, columns=["Index", "Date & Hour", "Issues"]).to_csv(output_file, index=False)

        print(f"Invalid data saved to {output_file}")
    
    else:
        print(f"No invalid data points found in {file_path}.")

    return invalid_data
   
base_folder = "./datasets/TX-Data"
sm_folder = os.path.join(base_folder, "soil_station")
met_folder = os.path.join(base_folder, "met_station")

# Call the function to process all weather data
process_all_weather_data(sm_folder, met_folder)
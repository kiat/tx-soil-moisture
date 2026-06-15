import pandas as pd

def check_for_NaT(df):
    """Check for NaT in the index and warn the user if there are any. Assumed: format=%Y-%m-%d %H:%M:%S """
    num_NaT = df.index.isna().sum()
    if num_NaT > 0:
        total_rows = len(df)
        percent_NaT = (num_NaT / total_rows) * 100
        print(f"Warning: {num_NaT} rows ({percent_NaT:.2f}%) have unparsable timestamps that were set to NaT. This may indicate a problem with the data or the timestamp format.")

def read_soil(file):
    """
    Read in soil data from a .dat file from the TxSON network, assuming format found in "TxSON_data_2026-02-24" and return a DataFrame with a datetimeindex.
    """
    df = pd.read_csv(file, header=5, dtype = str)

    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d %H:%M:%S")

    df = df.set_index('Date')

    # if there are NaT in the index, warn the user. 
    check_for_NaT(df)

    # THERE ARE SOME FILES WITH OUT-OF-ORDER DATES. 
    df = df.sort_index(kind='stable') # Retains original order of duplicate indices.

    # convert all but "Flag" column to numeric, coercing errors to NaN "
    for col in df.columns:
        if col != "Flag":
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.columns = df.columns.str.strip()

    return df

def read_met(file):
    """
    Read in met data from a .dat file from the TxSON network, assuming format found in "TxSON_data_2026-02-24" and return a DataFrame with a datetimeindex.
    """
    
    df = pd.read_csv(file, header = 6, dtype = str)

    df.columns = df.columns.str.strip()

    df = df.drop(df.index[0:2]) # drops units and measurment types

    df = df.drop(columns=['RECORD'])

    # standardize the column names.
    rename_dict = {
        'TIMESTAMP': 'Date',
        'Rain_mm_Tot': 'Ppt',
        'AirTC_Avg': 'Tair',
        'WS_ms_S_WVT': 'Wind speed',
        'WindDir_D1_WVT': 'Wind direction',
        'SlrW_Avg': 'Srad'
    }

    df = df.rename(columns=rename_dict)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("Date")

    # if there are NaT in the index, warn the user. 
    check_for_NaT(df)

    # THERE ARE SOME FILES WITH OUT-OF-ORDER DATES. 
    df = df.sort_index(kind='stable') # Retains original order of duplicate indices.

    # convert all columns to numeric.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def file_to_indexed_df(file_path, is_soil_or_met = 'unknown'):
    """
    Given a file path and whether the file is soil or met data, read in the file and return a dataframe with a datetime index. If the type of data is unknown, return None.

    Assumes the file is in the format found in "TxSON_data_2026-02-24."
    """

    if file_path == "": 
        print("Warning: empty file path provided, skipping...")
        return None

    elif file_path.endswith(".dat") == False:
        print(f"Warning: {file_path} does not end with .dat, skipping...")
        return None

    elif is_soil_or_met == 'unknown':
        print("The type of data (soil or met) is unknown.")
        print(f"skipping reading {file_path}...")
        return None

    elif is_soil_or_met == 'soil':
        return read_soil(file_path)

    elif is_soil_or_met == 'met':
        return read_met(file_path)

    else:
        print(f"Error with {file_path}, unknown cause.")

    
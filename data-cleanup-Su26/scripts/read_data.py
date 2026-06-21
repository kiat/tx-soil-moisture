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
    # read measurements as native numerics (fast); keep only "Flag" as a string.
    # low_memory=False parses each column in one pass (avoids mixed-type warnings).
    df = pd.read_csv(file, header=5, skipinitialspace=True, dtype={"Flag": str}, low_memory=False)

    df.columns = df.columns.str.strip()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d %H:%M:%S")

    df = df.set_index('Date')

    # if there are NaT in the index, warn the user.
    check_for_NaT(df)

    # THERE ARE SOME FILES WITH OUT-OF-ORDER DATES.
    df = df.sort_index(kind='stable') # Retains original order of duplicate indices.

    # coerce any stray non-numeric measurement values to NaN ("Flag" is left as a string)
    meas_cols = df.columns.difference(["Flag"])
    df[meas_cols] = df[meas_cols].apply(pd.to_numeric, errors='coerce')

    return df

def read_met(file):
    """
    Read in met data from a .dat file from the TxSON network, assuming format found in "TxSON_data_2026-02-24" and return a DataFrame with a datetimeindex.
    """
    
    # skip the units & measurement-type rows (the 2 rows after the header) so the
    # measurement columns parse as native numerics; drop RECORD at read time.
    df = pd.read_csv(file, header=6, skiprows=[7, 8], skipinitialspace=True,
                     usecols=lambda c: c.strip() != "RECORD", low_memory=False)

    df.columns = df.columns.str.strip()

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

    # coerce any stray non-numeric values to NaN.
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def file_to_indexed_df(file_path, is_soil_or_met = 'unknown'):
    """
    Given a file path and whether the file is soil or met data, read in the file and return a dataframe with a datetime index. If the type of data is unknown, return None.

    Assumes the file is in the format found in "TxSON_data_2026-02-24."
    """

    if file_path == "": 
        print("Warning: empty file path provided, skipping...")
        return None

    elif not file_path.endswith(".dat"):
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
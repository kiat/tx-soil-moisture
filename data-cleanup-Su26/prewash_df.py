import pandas as pd
from datetime import datetime
import argparse
import os

expected_soil_header = "Date,Ppt,SWC_5,SWC_10,SWC_20,SWC_50,T_5,T_10,T_20,T_50,Flag"
expected_met_header = 'TIMESTAMP,RECORD,Rain_mm_Tot,AirTC_Avg,RH,WS_ms_S_WVT,WindDir_D1_WVT,SlrW_Avg,ETos,Rso'

# minimum header-column matches needed to classify a file. soil is 9 (not 11)
# because some stations only report 3 depths, omitting the SWC_50/T_50 columns.
min_soil_features = 9
min_met_features = 10

def get_n_lines(file_path, n):
    lines = []
    with open(file_path, 'r') as in_file:
        for line in in_file:
            lines.append(line)
            if len(lines) >= n:
                break
    return lines

def is_valid_date(date_str, format_str):
    """ Given a string and a date format:
    1. Check if the string is a valid date in the given format
    2. RETURN True if it is a valid date, False otherwise
    """
    try:
        return bool(datetime.strptime(date_str, format_str))
    except ValueError:
        return False

def find_timestamp(n_lines):
    """ Given a certain number of lines:
    1. check the lines for a timestamp
    2. RETURN True if there is a timestamp, False otherwise
    """
    for line in n_lines:
        entries = line.split(",")
        for entry in entries:
            if entry:
                entry = entry.strip('"\n') # drop surrounding double-quotes and any \n
                if is_valid_date(entry, "%Y-%m-%d %H:%M:%S"):
                    return True
    return False

def determine_data_file(file_path):
    """ Given a file:
    1. read the first 10 lines of the file
    2. check for the presence of a timestamp in the first 10 lines
    3. count the number of features in the user supplied current met and soil headers
    4. check for the presence of the features in the first 10 lines of the file
    5. RETURN the data type (soil, met, or unknown) based on the number of specific features found and the presence of a timestamp
    """

    # 1. make sure the file contains measurement data by checking for timestamp within the first 10 rows

    # get the first 10 lines
    lines = get_n_lines(file_path, 10)

    # get the lines with commas:
    rows = []
    for line in lines:
        if "," in line:
            rows.append(line)

    # check for YYYY-MM-DD hh:mm:ss in the rows
    if not find_timestamp(rows):
        return 'unknown'

    # 2. check for the features.

    # find the number of soil and met features in the files
    for row in rows:
        entries = row.split(",")

        soil_features = []
        met_features = []
        for entry in entries:
            if entry:
                entry = entry.strip('"\n') # drop surrounding double-quotes and any \n
                entry = entry.strip("'\n") # drop surrounding single-quotes and any \n
                entry = entry.strip() # drop any leading or trailing whitespace

                if entry in expected_soil_header:
                    soil_features.append(entry)

                if entry in expected_met_header:
                    met_features.append(entry)

        # determine type
        if len(soil_features) >= min_soil_features:
            return 'soil'
        elif len(met_features) >= min_met_features:
            return 'met'

    return 'unknown'

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
        return None

def treat_subhourly_data(df, precip_col="Ppt"):
    """
    Condense sub-hourly measurements (timestamps with non-zero minutes/seconds)
    onto the hourly grid so that every record lands on a whole hour.

    Some stations occasionally log a record off the hour (e.g. 18:45). This
    function folds those stray records into the hourly series using the
    following rule:

      * Precipitation (`precip_col`, default "Ppt") logged on a sub-hourly
        timestamp is SUMMED and added to the NEXT on-the-hour timestamp
        (e.g. rain logged at 18:15 and 18:45 is added to the 19:00 total),
        so no accumulated rainfall is lost.
      * Every other measurement on a sub-hourly timestamp is DISCARDED; the
        reading already recorded on the whole hour (e.g. the "Flag" column and
        all sensor values) is kept as-is.
      * If the receiving whole hour has no record yet, a new row is created
        carrying only the summed precipitation. Its other columns are left as
        NaN for downstream imputation to handle.
    """

    # a timestamp is sub-hourly if it does not sit exactly on the hour
    on_hour = df.index == df.index.floor("h")

    # nothing to condense
    if on_hour.all():
        return df

    sub = df[~on_hour]          # the stray, off-the-hour records
    hourly = df[on_hour].copy()  # the records already on a whole hour

    # carry sub-hourly precipitation forward to the next whole hour
    if precip_col in df.columns:
        # next on-the-hour timestamp for each sub-hourly row (18:45 -> 19:00)
        target_hours = sub.index.ceil("h")

        # sum the precip falling into each receiving hour
        # (min_count=1 keeps all-NaN groups as NaN instead of 0)
        carry = sub[precip_col].groupby(target_hours).sum(min_count=1).dropna()

        if not carry.empty:
            # make sure every receiving hour exists as a row
            missing = carry.index.difference(hourly.index)
            if len(missing) > 0:
                hourly = hourly.reindex(hourly.index.union(missing))

            # add the carried precip onto the hourly precip.
            # only touch hours that actually receive a contribution, treating a
            # missing/NaN hourly precip as 0 so the rainfall is not lost.
            add = carry.reindex(hourly.index)
            mask = add.notna()
            hourly.loc[mask, precip_col] = (
                hourly.loc[mask, precip_col].fillna(0) + add[mask]
            )

    # the sub-hourly rows themselves are dropped; only whole-hour rows remain
    return hourly.sort_index(kind="stable")

def fill_missing_timestamps(df):
    """
    expects a DatetimeIndexed df and returns a DatetimeIndexed df with all missing
    timestamps filled in with NaN measurements.
    """
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='h')
    full_range.name = df.index.name # preserve 'Date'
    return df.reindex(full_range)

def dup_cleaner(df):
    """
    Expects a DataFrame with a timestamp index (and, for soil data, a "Flag" column).

    Treats the following cases of duplicate timestamps, always keeping the first
    occurrence in the data's original order:
    1. same timestamp, measurements, and flag
    2. same timestamp and measurements, but different flag
    3. same timestamp, different measurements

    Returns a DataFrame with duplicate timestamps removed and the timestamp set
    as the index.
    """

    index_name = df.index.name

    # reset the index so the timestamp is included in the duplicate comparison
    # (drop_duplicates defaults to keep='first')

    ### Case 1: identical rows (timestamp, measurements, and flag) ###
    df = df.reset_index().drop_duplicates()

    ### Case 2: same timestamp and measurements, differing only in flag ###
    if "Flag" in df.columns:
        non_flag_cols = [col for col in df.columns if col != "Flag"]
        df = df.drop_duplicates(subset=non_flag_cols)

    ### Case 3: same timestamp, different measurements ###
    df = df.drop_duplicates(subset=index_name)

    df = df.set_index(index_name)

    return df

def find_and_replace_wrong_data(dfc):
    """
    Replaces values outside valid physical bounds with NA for final export.
    CREDITS: HANNAH AND ZUN
    """
    for col in ['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']:
        if col in dfc.columns:
            dfc.loc[(dfc[col] < 0) | (dfc[col] > 0.6), col] = pd.NA
    if 'Ppt' in dfc.columns:
        dfc.loc[dfc['Ppt'] < 0, 'Ppt'] = pd.NA
    if 'RH' in dfc.columns:
        dfc.loc[(dfc['RH'] < 0) | (dfc['RH'] > 100), 'RH'] = pd.NA
    if 'Wind speed' in dfc.columns:
        dfc.loc[(dfc['Wind speed'] < 0) | (dfc['Wind speed'] > 25), 'Wind speed'] = pd.NA
    if 'Wind direction' in dfc.columns:
        dfc.loc[(dfc['Wind direction'] < 0) | (dfc['Wind direction'] > 360), 'Wind direction'] = pd.NA
    if 'Srad' in dfc.columns:
        dfc.loc[dfc['Srad'] < 0, 'Srad'] = pd.NA
    for col in ['T_5', 'T_10', 'T_20', 'T_50', 'Tair']:
        if col in dfc.columns:
            dfc.loc[(dfc[col] < -30) | (dfc[col] > 60), col] = pd.NA
    return dfc

def diff_report(before, after):
    """
    Build a record of every row touched by a prewash step, grouped by timestamp.

    Rows are matched on their full contents (timestamp included) and multiplicity
    is preserved, so each row is tagged:
      * "removed" - a dropped row (e.g. a duplicate or a folded sub-hourly record);
      * "added"   - an inserted row (e.g. a filled gap). An edited row appears as
        the old row "removed" and the new row "added";
      * "kept"    - a row that survived unchanged at a timestamp where something
        else changed (e.g. the surviving copy of a de-duplicated timestamp).
    Only timestamps with at least one removed/added row are reported, and all rows
    sharing a timestamp are kept together so removed and kept can be compared.
    """
    idx_name = before.index.name or "index"
    b = before.reset_index()
    a = after.reset_index()

    # an occurrence counter makes otherwise-identical rows distinguishable, so
    # the comparison respects how many copies of a row were added or removed
    b["_occ"] = b.groupby(list(b.columns), dropna=False).cumcount()
    a["_occ"] = a.groupby(list(a.columns), dropna=False).cumcount()

    merged = b.merge(a, how="outer", indicator=True)

    # keep every row at any timestamp that saw a change, including its survivors
    changed_ts = merged.loc[merged["_merge"] != "both", idx_name].unique()
    report = merged[merged[idx_name].isin(changed_ts)].copy()

    # within a timestamp: survivor first, then what was removed, then what was added
    report["change"] = pd.Categorical(
        report["_merge"].map({"both": "kept", "left_only": "removed", "right_only": "added"}),
        categories=["kept", "removed", "added"], ordered=True,
    )
    report = report.drop(columns=["_occ", "_merge"])
    return report.sort_values([idx_name, "change"]).reset_index(drop=True)


def step_report_path(output_filepath, step_name):
    """Derive a report path from the output path, e.g. out.csv -> out_<step>_report.csv"""
    stem, ext = os.path.splitext(output_filepath)
    return f"{stem}_{step_name}_report{ext}"


def run_prewash_step(step, df, output_filepath):
    """Run one prewash step and write a CSV report of the rows it changed."""
    before = df.copy()
    after = step(df)
    report = diff_report(before, after)
    report.to_csv(step_report_path(output_filepath, step.__name__), index=False)
    return after

def prewash_df(file_path, output_filepath=None):
    """
    Prewash a TxSON .dat file: parse, sort, de-duplicate, snap to an hourly grid,
    and replace out-of-range values with NA.

    Returns a DataFrame with a datetime index.
    """

    data_type = determine_data_file(file_path)

    df = file_to_indexed_df(file_path, data_type)

    if df is None:
        print(f"Could not read {file_path} as TxSON {data_type} data; no output written.")
        return None

    # each step writes a CSV report of the rows it changed, named after the step
    prewash_steps = (dup_cleaner, treat_subhourly_data,
                     fill_missing_timestamps, find_and_replace_wrong_data)
    for step in prewash_steps:
        df = run_prewash_step(step, df, output_filepath)

    return df

def main():

    parser = argparse.ArgumentParser(
        description="Prewash a TxSON .dat file: parse, sort, de-duplicate, snap to an "
                    "hourly grid, and replace out-of-range values with NA."
    )

    parser.add_argument("input_file", help="raw .dat file to prewash")
    parser.add_argument("output_filepath", nargs="?", default=None,
                        help="path to write the prewashed CSV "
                             "(default: <input_name>_prewashed.csv in the current directory)")

    args = parser.parse_args()

    # input validation
    if args.input_file.strip() == "":
        raise ValueError("input_file must be a non-empty string.")

    if not args.output_filepath or args.output_filepath.strip() == "":
        stem = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_filepath = os.path.join(os.getcwd(), f"{stem}_prewashed.csv")
        print(f"no output_filepath provided, writing to {args.output_filepath}")

    data_type = determine_data_file(args.input_file)

    df = file_to_indexed_df(args.input_file, data_type)

    if df is None:
        print(f"Could not read {args.input_file} as TxSON {data_type} data; no output written.")
        return

    # each step writes a CSV report of the rows it changed, named after the step
    prewash_steps = (dup_cleaner, treat_subhourly_data,
                     fill_missing_timestamps, find_and_replace_wrong_data)
    for step in prewash_steps:
        df = run_prewash_step(step, df, args.output_filepath)

    df.to_csv(args.output_filepath)

if __name__ == "__main__":
    main()
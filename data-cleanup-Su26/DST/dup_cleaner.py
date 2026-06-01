import pandas as pd

def dup_cleaner(df):
    """
    Expects a DataFrame with a timestamp index and a "Flag" column, drop the record column in met data.

    treats the following cases of duplicate timestamps:
    1. same timestamp, measurements, and flag -> drop duplicates, keep one of them (default behavior of drop_duplicates)
    2. same timestamp, measurements, but different flag -> set the flag to NA

    LEAVES SAME TIMESTAMP AND DIFFERENT MEASUREMENTS TO BE RESOLVED IN time_cleaner.py

    Returns a DataFrame with duplicate timestamps handled according to the above rules, and the timestamp set as the index.
    """


    # make sure the time stamp is the index,
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("timestamp column must be set as index before running dup_cleaner.py.")
    
    ### Case 1 ###

    # first reset index so that drop duplicates includes the timestamp column
    index_name = df.index.name
    df = df.reset_index().drop_duplicates() # default is keep='first', which is what we want for case 1 (keep one of the duplicates and drop the rest)

    
    ### Case 2 ###

    # for this case, make the flag Na if there are duplicates with the same timestamp and measurements but different flags.
    if "Flag" in df.columns:

        # find the measurements column by looking for the column that is not the flag column
        measurements_col = df.columns.difference(["Flag"])

        subset_cols = [index_name] + list(measurements_col)

        # set the flag to NA for just duplicates with the same timestamp and measurements but different flags
        conflicts = df.duplicated(subset=subset_cols, keep=False)
        df.loc[conflicts, "Flag"] = pd.NA

    # set the timestamp back as the index
    df = df.set_index(index_name)

    return df



    

    
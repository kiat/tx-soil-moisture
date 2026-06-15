import pandas as pd

def dup_cleaner(df):
    """
    Expects a DataFrame with a timestamp index and a "Flag" column, drop the record column in met data.

    treats the following cases of duplicate timestamps:
    1. same timestamp, measurements, and flag -> keep the first occurance by the original order of the data
    2. same timestamp, measurements, but different flag -> keep the first occurance by the original order of the data
    3. same timestamp, different measurements -> keep first occurance by the original order of the data

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

    # for this case, keep the first occurance by the original order of the data
    if "Flag" in df.columns:
        non_flag_cols = [col for col in df.columns if col != "Flag"] # get all columns except the flag column
        df = df.drop_duplicates(subset=non_flag_cols, keep='first') # drop duplicates based on non-flag columns, keeping the first occurance (which is the one with the first flag value in the original order)

    ### Case 3 ###

    # for this case, keep the first occurance by the original order of the data
    df = df.drop_duplicates(subset=index_name, keep='first') # drop duplicates based

    # set the timestamp back as the index
    df = df.set_index(index_name)

    return df



    

    
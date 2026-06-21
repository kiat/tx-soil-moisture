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
    if df.empty:
        return df

    ### Case 1 ###

    # first reset index so that drop duplicates includes the timestamp column
    index_name = df.index.name
    
    df = df.reset_index().drop_duplicates() # default is keep='first' (keep the first of the duplicates and drop the rest)

    ### Case 2 ###

    # for this case, keep the first occurance by the original order of the data
    if "Flag" in df.columns:
        non_flag_cols = [col for col in df.columns if col != "Flag"] # get all columns except the flag column
        df = df.drop_duplicates(subset=non_flag_cols) # drop duplicates based on non-flag columns

    ### Case 3 ###

    # for this case, keep the first occurance by the original order of the data
    df = df.drop_duplicates(subset=index_name) # drop duplicates based

    # set the timestamp back as the index
    df = df.set_index(index_name)

    return df

def main():
    from soil_or_met import SoilOrMet
    from read_data import file_to_indexed_df
    import argparse

    classifier = SoilOrMet() 

    parser = argparse.ArgumentParser(
        description="Drop duplicate timestamps from a TxSON .dat file."
    )

    parser.add_argument("input_file",  help="raw .dat file to clean")
    parser.add_argument("output_file", help="path to write the cleaned CSV")

    args = parser.parse_args()

    data_type = classifier.determine_data_file(args.input_file)

    df = file_to_indexed_df(args.input_file, data_type)

    if df is None:
        print(f"Failed to read {args.input_file} into a dataframe, but was identified as {data_type} data.")
        print( "skipping both duplicate cleaning and writing to csv...")
        return
    
    df = dup_cleaner(df)
    df.to_csv(args.output_file)

    print(f"\ndup clean data written to {args.output_file}\n")

if __name__ == "__main__":
    main()


    

    
import pandas as pd
 
def time_cleaner(df):
    """
    expects a DatetimeIndexed df and returns a DatetimeIndexed df with all missing 
    timestamps filled in with NaN measurements.
    """
    
    # make sure the index is the timestamps
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex (run dup_cleaner.py first).")
    if df.empty:
        return df
 
    # --- case 1: insert missing timestamps ---
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='h')
    full_range.name = df.index.name           # preserve 'Date'
    return df.reindex(full_range)

def main():
    from soil_or_met import SoilOrMet
    from read_data import file_to_indexed_df
    from dup_cleaner import dup_cleaner
    from gap_report import gap_report
    import argparse

    classifier = SoilOrMet() 

    parser = argparse.ArgumentParser(
        description="Fill in missing timestamps with NaN measurements in a TxSON .dat file."
    )

    parser.add_argument("input_file",  help="raw .dat file to clean")
    parser.add_argument("output_file", help="path to write the cleaned CSV")

    args = parser.parse_args()

    data_type = classifier.determine_data_file(args.input_file)

    df = file_to_indexed_df(args.input_file, data_type)

    # must clean dups
    df = dup_cleaner(df)

    print("\ngap report for the original data:")
    gap_report(df)

    # clean the timestamps
    df_cleaned = time_cleaner(df)

    # return if there is any change to the timestamps
    if df_cleaned.index.equals(df.index):
        print("No missing timestamps found, no changes made to the timestamps.")
        return

    print("\nThere are missing timestamps in the data. filling in with NaN measurements.")

    # report the groups of gaps in the timestamps
    print("\ngap report for the cleaned data:")
    gap_report(df_cleaned)

    # save the cleaned DataFrame to a CSV file
    df_cleaned.to_csv(args.output_file)

if __name__ == "__main__":
    main()

    
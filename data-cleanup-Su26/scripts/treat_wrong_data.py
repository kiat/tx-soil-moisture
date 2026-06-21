import pandas as pd

def find_and_replace_wrong_data(dfc):
    """
    Replaces values outside valid bounds with NaN for final export.
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

def main():
    from soil_or_met import SoilOrMet
    from read_data import file_to_indexed_df
    import argparse

    classifier = SoilOrMet()

    parser = argparse.ArgumentParser(
        description="Replace out-of-range measurement values with NA in a TxSON .dat file."
    )

    parser.add_argument("input_file",  help="raw .dat file to clean")
    parser.add_argument("output_file", help="path to write the cleaned CSV")

    args = parser.parse_args()

    data_type = classifier.determine_data_file(args.input_file)

    df = file_to_indexed_df(args.input_file, data_type)

    if df is None:
        print(f"Failed to read {args.input_file} into a dataframe, but was identified as {data_type} data.")
        print( "skipping bad-value treatment and writing to csv...")
        return

    # count how many values get replaced (NA before vs after)
    na_before = int(df.isna().sum().sum())

    df = find_and_replace_wrong_data(df)

    na_after = int(df.isna().sum().sum())
    print(f"Replaced {na_after - na_before} out-of-range value(s) with NA.")

    df.to_csv(args.output_file)

    print(f"\ncleaned data written to {args.output_file}\n")


if __name__ == "__main__":
    main()
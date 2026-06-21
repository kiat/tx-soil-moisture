import pandas as pd


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
        NaN for time_cleaner.py / downstream imputation to handle.

    Expects a DatetimeIndexed df. Run dup_cleaner.py first (so each hour is a
    unique timestamp) and time_cleaner.py after (to fill any remaining gaps).

    Returns a DatetimeIndexed df whose index contains only whole-hour timestamps.
    """

    # make sure the index is the timestamps
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex (run dup_cleaner.py first).")
    if df.empty:
        return df

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


def main():
    from soil_or_met import SoilOrMet
    from read_data import file_to_indexed_df
    from dup_cleaner import dup_cleaner
    import argparse

    classifier = SoilOrMet()

    parser = argparse.ArgumentParser(
        description="Condense sub-hourly measurements onto the hourly grid in a TxSON .dat file."
    )

    parser.add_argument("input_file",  help="raw .dat file to clean")
    parser.add_argument("output_file", help="path to write the cleaned CSV")

    args = parser.parse_args()

    data_type = classifier.determine_data_file(args.input_file)

    df = file_to_indexed_df(args.input_file, data_type)

    if df is None:
        print(f"Failed to read {args.input_file} into a dataframe, but was identified as {data_type} data.")
        print( "skipping both sub-hourly treatment and writing to csv...")
        return

    # resolve duplicate timestamps first so each hour is a unique timestamp
    df = dup_cleaner(df)

    # count the stray, off-the-hour records before treating them
    n_sub = int((df.index != df.index.floor("h")).sum())

    df_treated = treat_subhourly_data(df)

    if n_sub == 0:
        print("No sub-hourly timestamps found, no changes made.")
    else:
        print(f"Condensed {n_sub} sub-hourly timestamp(s) onto the hourly grid.")
        print(f"rows before: {len(df)}, after: {len(df_treated)}")

    df_treated.to_csv(args.output_file)

    print(f"\nsub-hourly treated data written to {args.output_file}\n")


if __name__ == "__main__":
    main()

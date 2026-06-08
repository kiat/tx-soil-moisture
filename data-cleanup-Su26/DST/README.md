# Notebooks

### DST-Check.ipynb

- This is a notebook that walks you through checking whether the TxSON network observes daylight savings or not and the quirks of the data found.

- **Reference this first** as the rest of the files where created in response to what is found.

###  demo.ipynb

- showcases the use of time_cleaner.py and dup_cleaner.py. The output is a fully duplicate free dataframe with missing timestamps (within the range of the original data file) filled in with Na. 

###  finding_dups.ipynb

- This notebook walks through the development of tools to get the files in the duplicate_report. Change the file path at the top to the path of the 'TxSON_data_2026-02-24' folder on your local machine and run all to get the full report.

- Reference README.md in the duplicate report folder for in depth explanation of the report and duplicates.

- important notes:

    - **The index is reset first.** `drop_duplicates` and `duplicated` look only at
    columns, never the index. The pipeline calls `reset_index()` so the timestamp
    becomes a column and participates in duplicate detection. This is correct when
    the index is a meaningful key such as a timestamp with possible repeats. It
    would be *wrong* if the index were a plain `RangeIndex`, because that injects a
    unique `0..n-1` column and then no row can ever be a duplicate.
    - **Missing values match each other.** Both functions inherit pandas' rule that
    `NaN`, `NaT`, `None`, and `pd.NA` are treated as equal to themselves when
    comparing rows. Two rows that are both missing in the same column (and equal
    elsewhere) count as duplicates rather than being skipped.
    - **Order matters.** Case two consumes case one's *de-duplicated* output. Running
    it on the raw frame instead would fold the exact duplicates in with the flag
    conflicts and inflate the case-two counts.


# Scripts

### soil_or_met.py

- This is a python script that has the tools to return whether a data file pertains to soil measurements, weather measurements, or is unknown.

### dup_cleaner.py

- Expects a DataFrame with a timestamp index and a "Flag" column, drop the record column in met data.

    - Treats the following cases of duplicate timestamps:
        1. same timestamp, measurements, and flag -> drop duplicates, keep one of them (default behavior of drop_duplicates)
        2. same timestamp, measurements, but different flag -> set the flag to NA

    - LEAVES SAME TIMESTAMP AND DIFFERENT MEASUREMENTS TO BE RESOLVED IN time_cleaner.py

    - Returns a DataFrame with majority of duplicate timestamps handled according to the above rules, and the timestamp set as the index.

### time_cleaner.py

- Expects a df with a DatetimeIndex and with 'RECORD' and 'Flag' columns dropped.
 
    - Two cases are handled:
        - Gaps:
            - missing timestamps are inserted on a regular hourly grid (new rows are left as NaN).
        - Collisions:
            - rows that share a timestamp but hold different measurements. Exact duplicates are assumed to have already been removed by dup_cleaner.py.
 
    - A collision is only resolved if a strategy is chosen:
        - avg_diff_measurements (default false): 
            - collapse the colliding rows into one averaged row.
        - extend_time (default false):           
            - keep the row closest to the previous measurement at the original time, push the rest into the following slots, and shift later rows forward so nothing overlaps.

### read_data.py

- contains: check_for_NaT, read_soil, read_met, and file_to_indexed_df.

- **file_to_indexed_df** calls the rest after you provide a file path to the raw dat file and specify if it is a 'soil' or 'met' file (ref soil_or_met.py file for automation of this). Returns a dataframe with the timestamps as the index.

- **read_soil**
    - assuming a soil file is provided like CB01.dat found in TxSON_data_2026-02-24, the method will:
        1. read in csv skipping first 5 lines
        2. read all columns in as strings 
        3. convert 'Date' column to datetime object, format must be "%Y-%m-%d %H:%M:%S", else coerced into NaT
        4. makes the 'Date' column the index
        5. checks for any NaT in the index and warns the user if so.
        6. sorts the index ascendingly
        7. converts all columns to numeric, coercing errors to NaN, **except** Flag column.
        8. Returns the dataframe

- **read_met**
    - assuming a met file is provided like CB01_met.dat found in TxSON_data_2026-02-24, the method will:
        1. read in csv skipping first 6 lines
        2. read all columns in as strings
        3. drop the first two rows
        4. drop the 'RECORD' column
        5. convert the 'TIMESTAMP' column to datetime object, format must be "%Y-%m-%d %H:%M:%S", else coerced into NaT
        6. makes the 'TIMESTAMP' column the index
        7. checks for any NaT in the index and warns the user if so.
        8. sorts the index ascendingly
        9. converts all columns to numeric, coercing errors to NaN
    

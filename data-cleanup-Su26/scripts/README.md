###  demo.ipynb

- showcases the use of all scripts.

# Scripts

## 1. soil_or_met.py

- This is a python script that has the tools to return whether a data file pertains to soil measurements, weather measurements, or is unknown.

---
## 2. dup_cleaner.py

- Expects a DataFrame with a timestamp index and a "Flag" column, drop the record column in met data.

- Treats the following cases of duplicate timestamps:
    1. same timestamp, measurements, and flag -> keep first
    2. same timestamp, measurements, but different flag -> keep first
    3. same timestamp, different measurements and flags -> keep first

- Returns a DataFrame with all the cases above treated and with a DatetimeIndex

---
## 3. time_cleaner.py

- Expects a df with a DatetimeIndex 

- Returns a DatetimeIndexed df with missing hourly timestamps filled in with NA measuremnts

---
## 4. read_data.py

- contains: check_for_NaT, read_soil, read_met, and file_to_indexed_df.

- **file_to_indexed_df** calls the rest after you provide **a file path** to the raw dat file and specify if it is a 'soil' or 'met' file (ref soil_or_met.py file for automation of this). Returns a dataframe with the timestamps as the index.

- **read_soil**
    - assuming a soil file is provided like CB01.dat found in TxSON_data_2026-02-24, the method will:
        1. read in csv skipping first 5 lines
        2. read all columns in as strings 
        3. convert 'Date' column to datetime object, format must be "%Y-%m-%d %H:%M:%S", else coerced into NaT
        4. makes the 'Date' column the index
        5. checks for any NaT in the index and warns the user if so.
        6. sorts stabely the index ascendingly
        7. converts all columns to numeric, coercing errors to NaN, **except** Flag column.
        8. Returns the DatetimeIndexed dataframe

- **read_met**
    - assuming a met file is provided like CB01_met.dat found in TxSON_data_2026-02-24, the method will:
        1. read in csv skipping first 6 lines
        2. read all columns in as strings
        3. drop the first two rows
        4. drop the 'RECORD' column
        5. **Standardize Column names**
        6. convert the 'TIMESTAMP' column to datetime object, format must be "%Y-%m-%d %H:%M:%S", else coerced into NaT
        7. makes the 'TIMESTAMP' column the index
        8. checks for any NaT in the index and warns the user if so.
        9. sorts stabely the index ascendingly
        10. converts all columns to numeric, coercing errors to NaN
        11. Returns the DatetimeIndexed dataframe

---
## 5. get_data_dict.py

- The full data ingest (read + prewash + clean) pipeline. Prewashing treats duplicates, missing timestamps, replaces impossible data with NA, and other oddities. Cleaning will impute the missing data.

---
## 6. treat_wrong_data.py

- Using hardcoded column and acceptable ranges, this will replace all data deemed impossible with NA. 

---
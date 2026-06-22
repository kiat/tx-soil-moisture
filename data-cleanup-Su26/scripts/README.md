###  demo.ipynb

- Showcases the use of all scripts, with the command-line usage for each.

# Scripts

Run these from inside the `scripts/` folder — the modules import one another by name. Every script except `soil_or_met.py` and `read_data.py` also has a CLI (shown below); the single-file tools take `<input.dat> <output.csv>`.

Prewash pipeline order: **dup_cleaner → treat_subhourly_data → time_cleaner → treat_wrong_data**.

---
## 1. soil_or_met.py  *(import only)*

- Tools to determine whether a data file pertains to soil measurements, weather (met) measurements, or is unknown.

---
## 2. read_data.py  *(import only)*

- contains: check_for_NaT, read_soil, read_met, and file_to_indexed_df.

- **file_to_indexed_df** calls the rest after you provide **a file path** to the raw .dat file and specify whether it is a 'soil' or 'met' file (see soil_or_met.py to automate this). Returns a DataFrame with the timestamps as the index.

- **read_soil** — assuming a soil file like CB01.dat, the method will:
    1. read the CSV skipping the first 5 lines, parsing the measurement columns as native numerics and keeping `Flag` as a string
    2. convert the `Date` column to a datetime (format `%Y-%m-%d %H:%M:%S`, else coerced to `NaT`) and set it as the index
    3. warn if any `NaT` ended up in the index
    4. sort the index stably (ascending)
    5. coerce any stray non-numeric measurement values to `NaN` (`Flag` stays a string)
    6. return the DatetimeIndexed DataFrame

- **read_met** — assuming a met file like CB01_met.dat, the method will:
    1. read the CSV using line 6 as the header, skipping the two units / measurement-type rows and dropping `RECORD` at read time
    2. standardize the column names (`TIMESTAMP`→`Date`, `Rain_mm_Tot`→`Ppt`, …)
    3. convert the `Date` column to a datetime and set it as the index
    4. warn if any `NaT` ended up in the index
    5. sort the index stably (ascending)
    6. coerce any stray non-numeric values to `NaN`
    7. return the DatetimeIndexed DataFrame

---
## 3. dup_cleaner.py

- Expects a DataFrame with a timestamp index and a "Flag" column.

- Treats the three cases of duplicate timestamps (all keep the first occurrence):
    1. same timestamp, measurements, and flag
    2. same timestamp, measurements, but different flag
    3. same timestamp, different measurements

- Returns a DataFrame with the duplicates resolved and a DatetimeIndex.

- **CLI:** `python dup_cleaner.py <input.dat> <output.csv>`

---
## 4. treat_subhourly_data.py

- Condenses sub-hourly readings (timestamps with non-zero minutes) onto the hourly grid: sub-hourly precipitation (`Ppt`) is summed into the next whole hour, every other measurement keeps the value already on the whole hour, and the sub-hourly rows are dropped. Run after dup_cleaner and before time_cleaner.

- **CLI:** `python treat_subhourly_data.py <input.dat> <output.csv>`

---
## 5. time_cleaner.py

- Expects a df with a unique, whole-hour DatetimeIndex.

- Returns a DatetimeIndexed df with the missing hourly timestamps filled in with `NaN` measurements.

- **CLI:** `python time_cleaner.py <input.dat> <output.csv>`

---
## 6. treat_wrong_data.py

- Using hardcoded columns and acceptable ranges, replaces all values deemed impossible with `NA` (the `Flag` column is left untouched).

- **CLI:** `python treat_wrong_data.py <input.dat> <output.csv>`

---
## 7. gap_report.py

- Tallies data gaps by duration (`<24h`, `1-7d`, `7-30d`, `>30d`) for both the timestamp index (missing hours) and each column (consecutive-`NaN` runs); prints and returns the table.

- **CLI:** `python gap_report.py <input.dat> [output.csv]`  (output CSV optional)

---
## 8. get_data_dict.py

- The full data ingest (read + prewash + clean) pipeline over a folder of raw files. Prewashing applies dup_cleaner → treat_subhourly_data → time_cleaner → treat_wrong_data to each station; cleaning (impute) is not yet implemented.

- **CLI:** `python get_data_dict.py <input_folder> --prewash --download [--prewash-folder DIR]`

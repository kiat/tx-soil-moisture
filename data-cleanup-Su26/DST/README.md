
# DST-Check.ipynb

- This is a notebook that walks you through checking whether the TxSON network observes daylight savings or not and the quirks of the data found.

- Reference this first as the rest of the files where created in response to what is found.

# soil_or_met.py

- This is a python script that has the tools to return whether a data file pertains to soil measurements, weather measurements, or is unknown.

# dup_cleaner.py

- Expects a DataFrame with a timestamp index and a "Flag" column, drop the record column in met data.

    - Treats the following cases of duplicate timestamps:
        1. same timestamp, measurements, and flag -> drop duplicates, keep one of them (default behavior of drop_duplicates)
        2. same timestamp, measurements, but different flag -> set the flag to NA

    - LEAVES SAME TIMESTAMP AND DIFFERENT MEASUREMENTS TO BE RESOLVED IN time_cleaner.py

    - Returns a DataFrame with majority of duplicate timestamps handled according to the above rules, and the timestamp set as the index.

# time_cleaner.py

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

# demo.ipynb

- showcase the use of time_cleaner.py and dup_cleaner.py
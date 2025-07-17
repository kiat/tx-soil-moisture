#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import timedelta
import os

import pandas as pd
import numpy as np
from datetime import timedelta

# Initialize station data
stations = [pd.read_csv(f"https://www.cs.utexas.edu/~kiat/datasets/Revised_Final_Data/Station{i}_Revised_Final_Data.csv") for i in range(1, 7)]
station_1, station_2, station_3, station_4, station_5, station_6 = stations

def detect_rainfall_events(df, dry_threshhold=6):
    result_df = df.copy()
    result_df['rain'] = (result_df['Ppt'] > 0.5).astype(int)
    wet_index = df.index[df['rain'] == 1].to_list()
    event_ids = np.full(len(df), -1)
    event_counter = 0
    if wet_index:
        event_ids[wet_index[0]] = event_counter
        for i in range(1, len(wet_index)):
            gap = wet_index[i] - wet_index[i - 1] - 1
            if gap > dry_threshhold:
                event_counter += 1
            event_ids[wet_index[i]] = event_counter
    result_df['event_id'] = event_ids
    return result_df

for data in stations:
    data = detect_rainfall_events(data)

def compute_swc_return_times(
        df: pd.DataFrame,
        buffer_hours: int = 2,
        depth_prefix: str = "SWC_"
) -> pd.DataFrame:
    """
    For every rainfall *event × soil-depth*, compute how many hours it takes
    for soil-water-content (SWC) to return to its pre-rain baseline.

    Parameters
    ----------
    df : DataFrame
        * Requirements:
          – DatetimeIndex (hourly)
          – 'event_id' column from `detect_rainfall_events`
          – one or more SWC columns whose names start with `depth_prefix`.
    buffer_hours : int, default 2
        Number of hours immediately **before** the event used to calculate
        the baseline SWC.
    depth_prefix : str, default "SWC_"
        Prefix identifying SWC columns (e.g. 'SWC_5', 'SWC_20', ...).

    Returns
    -------
    tidy : DataFrame
        Columns:
            event_id, depth_cm, start, end,
            baseline_swc, return_time_h  (NaN if never dries in record)
    """
    # ------------------------------------------------------------------
    # 1) Identify SWC columns
    depth_cols = [c for c in df.columns if c.startswith(depth_prefix)]
    if not depth_cols:
        raise ValueError(f"No columns start with '{{depth_prefix}}'")

    # ------------------------------------------------------------------
    # 2) Build a quick per-event summary (start & end timestamps)
    events = (
        df[df["event_id"] >= 0]              # keep only “wet” rows
        .groupby("event_id")
        .agg(start=("event_id", lambda x: x.index[0]),
             end  =("event_id", lambda x: x.index[-1]))
    )

    # ------------------------------------------------------------------
    # 3) Loop through events and depths
    records = []
    for ev_id, (ev_start, ev_end) in events.iterrows():
        # ---- baseline: mean SWC in the look-back window -------------
        pre_win = df.loc[ev_start - timedelta(hours=buffer_hours):
                         ev_start - timedelta(hours=1)]
        if pre_win.empty:                         # event at file start
            pre_win = df.loc[[ev_start]]
        baseline = pre_win[depth_cols].mean()

        # ---- scan forward until SWC ≤ baseline ----------------------
        post_rain = df.loc[ev_end + timedelta(hours=1):]
        for col in depth_cols:
            base_val  = baseline[col]
            rt_hours  = np.nan                    # default if never dries

            for ts, swc_val in post_rain[col].items():
                if swc_val <= base_val:
                    rt_hours = (ts - ev_end).total_seconds() / 3600.0
                    break

            records.append({{
                "event_id"      : ev_id,
                "depth_cm"      : int(col.split("_")[1]),
                "start"         : ev_start,
                "end"           : ev_end,
                "baseline_swc"  : round(base_val, 3),
                "return_time_h" : rt_hours,
            }})

    tidy = pd.DataFrame(records).sort_values(["event_id", "depth_cm"])
    return tidy


def determine_rain_impact(df, window = 3):
    """ Analyze how soil moisture changes in response to rainfall events 

        window : integer that determines the number of hours viewed before 
                 and after a rainfall event
    """
    var_list = ['SWC_5','SWC_10','SWC_20','SWC_50']
    
    avg_before_moisture = []
    avg_after_moisture = []
    result =[]

    for var in var_list:
        for i in range(len(df)):
            if df.iloc[i]['event_id'] != -1:
                for n in range(1,4):
                    if i+n <= len(df):
                        avg_after_moisture.append(df.iloc[i+n][var])
                for n in range(3,0,-1):
                    if i-n >= 0:
                        avg_before_moisture.append(df.iloc[i-n][var])
        
        result += (sum(avg_after_moisture) / len(avg_after_moisture)) - (sum(avg_before_moisture) / len(avg_before_moisture))
    
    return result

def find_lag(df, window = 24, station_number = 1, threshhold = 0):
    """ Determine the average number of hours it takes for a rainfall \
        event to affect the deeper soil levels (20 and 50cm)

        window : the number of hours to check after a rainfall event for change
        station_number : number of station
    """

    lag_hours_20 = []
    lag_hours_50 = []
    
    event_starts = df[df['event_id'] != -1].groupby('event_id').first().reset_index()
    event_starts['total_precipitation'] = df[df['event_id'] != 1].groupby('event_id')['Ppt'].sum()

    for i in range(len(event_starts)):
        baseline_index = df[df['event_id'] == i].index[0]
        baseline_swc_20 = df.loc[baseline_index, 'SWC_20']
        baseline_swc_50 = df.loc[baseline_index, 'SWC_50']

        for lag in range(1, window + 1):
            check_index = baseline_index + lag
            if check_index >= len(df):
                break

            swc_20 = df.loc[check_index, 'SWC_20']
    
            if (swc_20 != baseline_swc_20) and ((swc_20 > (baseline_swc_20 + threshhold)) or (swc_20 < (baseline_swc_20 - threshhold))):
                lag_hours_20.append(lag)
                break

        for lag in range(1, window + 1):
            check_index = baseline_index + lag
            if check_index >= len(df):
                break

            swc_50 = df.loc[check_index, 'SWC_50']

            if (swc_50 != baseline_swc_50) and ((swc_50 > (baseline_swc_50 + (threshhold/2))) or (swc_50 < (baseline_swc_50 - (threshhold/2)))):
                lag_hours_50.append(lag)
                break

    avg_lag_20 = sum(lag_hours_20) / df['event_id'].max()
    avg_lag_50 = sum(lag_hours_50) / df['event_id'].max()

    return {
        'station_id': int(station_number),
        'total_rainfall': sum(df['Ppt']),
        'avg_rainfall': sum(df['Ppt']) / df['event_id'].max(),
        'avg_lag_20': avg_lag_20,
        'avg_lag_50': avg_lag_50
    }
    
def validate_and_run():
    while True:
        file_path = input("Enter the path to the station CSV file: ").strip()
        if not os.path.isfile(file_path):
            print("File does not exist. Please try again.")
            continue

        try:
            df = pd.read_csv(file_path, parse_dates=["Timestamp"])
            df = df.set_index("Timestamp").sort_index()
            if "Ppt" not in df.columns or not any(col.startswith("SWC_") for col in df.columns):
                raise ValueError("Missing required column(s): 'Ppt' or SWC depth columns.")
        except Exception as e:
            print(f"Invalid file or format: {e}")
            continue

        df = detect_rainfall_events(df)
        summary_df = compute_swc_return_times(df)
        output_file = os.path.splitext(file_path)[0] + "_SWC_return_times.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"Output saved to: {output_file}")
        break

def swc_percentage_change(df, hour = 1):
    results = []

    event_starts = df[df['event_id'] != -1].groupby('event_id').first().reset_index()

    for _, row in event_starts.iterrows():
        event_id = row['event_id']
        event_start_idx = df[df['event_id'] == event_id].index[0]
        after_idx = event_start_idx + 1

        if after_idx >= len(df):
            continue

        baseline = df.loc[event_start_idx, 'SWC_10']
        after = df.loc[after_idx, 'SWC_10']

        percent_change = (after - baseline) / baseline if baseline != 0 else 0

        results.append({
            'event_id': event_id,
            'baseline': baseline,
            'after_1hr': after,
            'percent_change': percent_change
        })

    return pd.DataFrame(results)



if __name__ == "__main__":
    validate_and_run()

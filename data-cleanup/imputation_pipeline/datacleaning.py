"""Build merged, cleaned, and missing-summary CSVs for one station.

Examples:
    python datacleaning.py --station 1
    python datacleaning.py --station CB01 --soil-base-dir ../../datasets/TxSON_data_2026-02-24 --met-base-dir ../../datasets/TxSON_data_2026-02-24
"""

import argparse
import io
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Manual overrides for stations whose sensors reported bad-but-present readings
# that we want to treat as contiguous missing intervals.
MANUAL_GAP_RULES = {
    5: [
        {
            "parameters": ["T_20"],
            "start": "2015-01-01 00:00:00",
            "end": "2016-02-20 12:00:00",
        },
        {
            "parameters": ["T_5", "T_10", "T_20", "T_50"],
            "start": "2018-04-14 20:00:00",
            "end": "2018-05-15 08:00:00",
        },
    ]
}

SOIL_NUMERIC_COLUMNS = [
    "SWC_5", "SWC_10", "SWC_20", "SWC_50",
    "T_5", "T_10", "T_20", "T_50", "Ppt", "Flag",
]

MET_NUMERIC_COLUMNS = ["Ppt", "Tair", "RH", "Wind speed", "Wind direction", "Srad"]

TOA5_MET_RENAME = {
    "TIMESTAMP": "Date",
    "Rain_mm_Tot": "Ppt",
    "AirTC_Avg": "Tair",
    "WS_ms_S_WVT": "Wind speed",
    "WindDir_D1_WVT": "Wind direction",
    "SlrW_Avg": "Srad",
}


def station_key(station_id):
    return str(station_id).strip()


def manual_gap_key(station_id):
    key = station_key(station_id)
    return int(key) if key.isdigit() else key


def find_header_row(file_path, startswith):
    with Path(file_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_num, line in enumerate(handle):
            if line.startswith(startswith):
                return line_num
    return None


def resolve_station_file(base_dir, station_id, patterns, required=True):
    station = station_key(station_id)
    tried = []
    for pattern in patterns:
        candidate = Path(base_dir) / pattern.format(station=station)
        tried.append(str(candidate))
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"No station file found for {station}. Tried: {tried}")
    return None


def coerce_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate_observations_to_hourly(df):
    """Collapse sub-hourly observations to their hour without inventing empty hours."""
    if df.empty:
        return df
    aligned_to_hour = (
        (df.index.minute == 0)
        & (df.index.second == 0)
        & (df.index.microsecond == 0)
    ).all()
    if aligned_to_hour:
        return df

    hourly_index = df.index.floor("H")
    pieces = {}
    for col in df.columns:
        series = df[col]
        if col == "Ppt":
            pieces[col] = series.groupby(hourly_index).sum(min_count=1)
        elif pd.api.types.is_numeric_dtype(series):
            pieces[col] = series.groupby(hourly_index).mean()
        else:
            pieces[col] = series.groupby(hourly_index).last()
    return pd.DataFrame(pieces).sort_index()


def finalize_datetime_index(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "Date" not in df.columns:
        raise ValueError("Expected a Date column after parsing station data.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def repair_concatenated_toa5_records(text):
    # A few TOA5 files contain two records glued together with no newline.
    return re.sub(r'(?<=[0-9])"(?=\d{4}-\d{2}-\d{2} )', '\n"', text)


def load_soil_data(station_id, base_dir):
    """Load old SM_N.dat or new SITE.dat soil data."""
    file_path = resolve_station_file(
        base_dir,
        station_id,
        patterns=["SM_{station}.dat", "{station}.dat"],
    )
    header_row = find_header_row(file_path, "Date,")
    if header_row is None:
        raise ValueError(f"No Date header found in soil file: {file_path}")

    df = pd.read_csv(file_path, sep=",", skiprows=header_row)
    df = finalize_datetime_index(df)
    df = coerce_numeric(df, SOIL_NUMERIC_COLUMNS)
    return aggregate_observations_to_hourly(df)


def load_met_data(station_id, base_dir):
    """Load old MET_N.dat or new SITE_met.dat MET data.

    Missing MET files are allowed because most new 33-station files are soil-only.
    """
    file_path = resolve_station_file(
        base_dir,
        station_id,
        patterns=["MET_{station}.dat", "{station}_met.dat"],
        required=False,
    )
    if file_path is None:
        return pd.DataFrame()

    old_header_row = find_header_row(file_path, "Date,")
    if old_header_row is not None:
        df = pd.read_csv(file_path, sep=",", skiprows=old_header_row)
        df = finalize_datetime_index(df)
        df = coerce_numeric(df, MET_NUMERIC_COLUMNS)
        return aggregate_observations_to_hourly(df)

    toa5_header_row = find_header_row(file_path, '"TIMESTAMP"')
    if toa5_header_row is None:
        raise ValueError(f"No Date or TOA5 TIMESTAMP header found in MET file: {file_path}")

    text = repair_concatenated_toa5_records(
        file_path.read_text(encoding="utf-8", errors="replace")
    )
    df = pd.read_csv(io.StringIO(text), skiprows=toa5_header_row, low_memory=False)
    # TOA5 rows after the header are units and aggregation metadata.
    df = df.iloc[2:].copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns=TOA5_MET_RENAME)

    keep_cols = ["Date"] + [col for col in MET_NUMERIC_COLUMNS if col in df.columns]
    df = df[keep_cols]
    df = finalize_datetime_index(df)
    df = coerce_numeric(df, MET_NUMERIC_COLUMNS)
    return aggregate_observations_to_hourly(df)


def merge_raw_data(station_id, soil_base_dir, met_base_dir):
    """Merge soil and MET data, keeping MET precipitation when both exist."""
    df_soil = load_soil_data(station_id, soil_base_dir)
    df_met = load_met_data(station_id, met_base_dir)
    if df_met.empty:
        return df_soil

    merged = pd.merge(
        df_soil,
        df_met,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("_soil", "_met"),
    )
    if "Ppt_soil" in merged.columns and "Ppt_met" in merged.columns:
        merged["Ppt"] = merged["Ppt_met"].combine_first(merged["Ppt_soil"])
        merged.drop(columns=["Ppt_soil", "Ppt_met"], inplace=True)

    return merged


def save_merged_data(df, station_id, output_dir):
    """Save merged raw data to CSV; return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"raw_merged_station_{station_id}.csv"
    df.to_csv(out_path)
    return str(out_path)


def find_missing_data(df: pd.DataFrame) -> dict:
    """Return a dict of columns to timestamps where data is NaN."""
    missing = {}
    for col in df.columns:
        idx = df.index[df[col].isnull()]
        if not idx.empty:
            missing[col] = idx.tolist()
    return missing


def find_and_replace_wrong_data(df):
    """Replace out-of-range values with NaN and return invalid timestamps."""
    wrong = {}
    dfc = df.copy()

    for col in ["SWC_5", "SWC_10", "SWC_20", "SWC_50"]:
        if col in dfc.columns:
            bad = dfc.index[(dfc[col] < 0) | (dfc[col] > 0.6)]
            if not bad.empty:
                wrong[col] = bad.tolist()
                dfc.loc[bad, col] = np.nan

    if "Ppt" in dfc.columns:
        bad = dfc.index[dfc["Ppt"] < 0]
        if not bad.empty:
            wrong["Ppt"] = bad.tolist()
            dfc.loc[bad, "Ppt"] = np.nan

    if "RH" in dfc.columns:
        bad = dfc.index[(dfc["RH"] < 0) | (dfc["RH"] > 100)]
        if not bad.empty:
            wrong["RH"] = bad.tolist()
            dfc.loc[bad, "RH"] = np.nan

    if "Wind speed" in dfc.columns:
        bad = dfc.index[(dfc["Wind speed"] < 0) | (dfc["Wind speed"] > 25)]
        if not bad.empty:
            wrong["Wind speed"] = bad.tolist()
            dfc.loc[bad, "Wind speed"] = np.nan

    if "Wind direction" in dfc.columns:
        bad = dfc.index[(dfc["Wind direction"] < 0) | (dfc["Wind direction"] > 360)]
        if not bad.empty:
            wrong["Wind direction"] = bad.tolist()
            dfc.loc[bad, "Wind direction"] = np.nan

    if "Srad" in dfc.columns:
        bad = dfc.index[dfc["Srad"] < 0]
        if not bad.empty:
            wrong["Srad"] = bad.tolist()
            dfc.loc[bad, "Srad"] = np.nan

    for col in ["T_5", "T_10", "T_20", "T_50", "Tair"]:
        if col in dfc.columns:
            bad = dfc.index[(dfc[col] < -30) | (dfc[col] > 60)]
            if not bad.empty:
                wrong[col] = bad.tolist()
                dfc.loc[bad, col] = np.nan

    return dfc, wrong


def combine_nan_lists(missing, wrong):
    """Merge the two dicts of timestamp lists and sort each list."""
    combined = {}
    for key in set(missing) | set(wrong):
        combined[key] = sorted(missing.get(key, []) + wrong.get(key, []))
    return combined


def group_consecutive_dates(dates: list, freq: pd.Timedelta) -> list:
    """Group timestamps that are within 1.5 times freq of each other."""
    if not dates:
        return []
    groups, current = [], [dates[0]]
    for prev, curr in zip(dates, dates[1:]):
        if curr - prev <= freq * 1.5:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]
    groups.append(current)
    return groups


def create_missing_summary_df(info):
    """Build a summary DataFrame from merged NaN/invalid timestamp info."""
    rows = []
    for param, ts_list in info.items():
        for grp in group_consecutive_dates(sorted(ts_list), pd.Timedelta(hours=1)):
            rows.append({
                "Parameter": param,
                "Start Timestamp": grp[0],
                "End Timestamp": grp[-1],
                "Number Missing": len(grp),
            })
    return pd.DataFrame(
        rows,
        columns=["Parameter", "Start Timestamp", "End Timestamp", "Number Missing"],
    )


def inject_manual_gaps(station_id, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Append known bad stretches that should be treated as missing gaps."""
    rules = MANUAL_GAP_RULES.get(manual_gap_key(station_id))
    if not rules:
        return summary_df

    manual_rows = []
    df = summary_df.copy()
    for rule in rules:
        start = pd.Timestamp(rule["start"])
        end = pd.Timestamp(rule["end"])
        hours = int((end - start) / pd.Timedelta(hours=1)) + 1
        for param in rule["parameters"]:
            if not df.empty:
                overlap = (
                    (df["Parameter"] == param)
                    & (df["Start Timestamp"] <= end)
                    & (df["End Timestamp"] >= start)
                )
                df = df[~overlap]
            manual_rows.append({
                "Parameter": param,
                "Start Timestamp": start,
                "End Timestamp": end,
                "Number Missing": hours,
            })

    if manual_rows:
        df = pd.concat([df, pd.DataFrame(manual_rows)], ignore_index=True)
        df = df.sort_values(["Parameter", "Start Timestamp"]).reset_index(drop=True)
    return df


def build_hourly_cleaned_data(merged_df):
    full_idx = pd.date_range(start=merged_df.index.min(), end=merged_df.index.max(), freq="H")
    full_df = merged_df.reindex(full_idx)
    return find_and_replace_wrong_data(full_df)


def main():
    parser = argparse.ArgumentParser(
        description="Generate merged, missing/invalid summary, and cleaned full-timeline CSVs for a soil station."
    )
    parser.add_argument("--station", "-s", type=str, default="1", help="Station ID or site code, e.g. 1 or CB01.")
    parser.add_argument("--soil-base-dir", type=str, default="../../datasets/TX-Data/soil_station", help="Path to soil .dat files.")
    parser.add_argument("--met-base-dir", type=str, default="../../datasets/TX-Data/met_station", help="Path to MET .dat files.")
    parser.add_argument("--raw-output-dir", type=str, default="raw_merged_data", help="Directory for merged CSVs.")
    parser.add_argument("--missing-output", type=str, default=None, help="Filename for missing/invalid summary CSV.")
    parser.add_argument("--cleaned-output", type=str, default=None, help="Filename for cleaned full-timeline CSV.")
    args = parser.parse_args()

    station_id = args.station

    merged_df = merge_raw_data(station_id, args.soil_base_dir, args.met_base_dir)
    raw_path = save_merged_data(merged_df, station_id, args.raw_output_dir)
    print(f"Saved merged data to: {raw_path}")

    cleaned_df, wrong = build_hourly_cleaned_data(merged_df)
    missing = find_missing_data(cleaned_df)
    combined = combine_nan_lists(missing, wrong)
    summary_df = create_missing_summary_df(combined)
    summary_df = inject_manual_gaps(station_id, summary_df)

    miss_out = args.missing_output or f"missing_data/Station{station_id}_missing_data.csv"
    os.makedirs(Path(miss_out).parent, exist_ok=True)
    summary_df.to_csv(miss_out, index=False)
    print(f"Missing/invalid summary saved to: {miss_out}")

    clean_out = args.cleaned_output or f"cleaned_data/Station{station_id}_cleaned_data.csv"
    os.makedirs(Path(clean_out).parent, exist_ok=True)
    cleaned_df.to_csv(clean_out, na_rep="NaN")
    print(f"Cleaned full-timeline data saved to: {clean_out}")


if __name__ == "__main__":
    main()

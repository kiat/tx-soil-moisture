"""Build merged, cleaned, and missing-summary CSVs for one station.

Examples:
    python datacleaning.py --station 1
    python datacleaning.py --station CB01 --soil-base-dir ../../datasets/TxSON_data_2026-02-24 --met-base-dir ../../datasets/TxSON_data_2026-02-24
"""

import argparse
import hashlib
import io
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from time_index_utils import require_unique_datetime_index

warnings.filterwarnings("ignore")

DEFAULT_TXSON_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "TxSON_data_2026-02-24"
)


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

DUPLICATE_SUMMARY_COLUMNS = [
    "Station", "Source", "Duplicate Timestamp Groups", "Duplicate Input Rows",
    "Rows Removed", "Exact Duplicate Groups", "Complementary Groups",
    "Measurement Conflict Groups", "Flag-Only Conflict Groups",
]

DUPLICATE_CONFLICT_COLUMNS = [
    "Station", "Source", "Timestamp", "Conflict Type", "Row Count",
    "Conflicting Parameters", "Source Rows",
]


@dataclass
class DuplicateAudit:
    summaries: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    source_coverage: dict = field(default_factory=dict)


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
    require_unique_datetime_index(df, "source after duplicate resolution")
    df = df.copy()
    if "Ppt" in df:
        # Invalid rain must not cancel a valid sub-hour total or block Soil fallback.
        df["Ppt"] = df["Ppt"].where(np.isfinite(df["Ppt"]) & df["Ppt"].ge(0))
    if df.empty:
        return df
    aligned_to_hour = (
        (df.index.minute == 0)
        & (df.index.second == 0)
        & (df.index.microsecond == 0)
    ).all()
    if aligned_to_hour:
        return df

    hourly_index = df.index.floor("h")
    pieces = {}
    for col in df.columns:
        series = df[col]
        if col == "Ppt":
            pieces[col] = series.groupby(hourly_index).sum(min_count=1)
        elif col == "Flag":
            pieces[col] = series.groupby(hourly_index).last()
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
    return df.sort_index(kind="stable")


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def resolve_duplicate_timestamps(
    df: pd.DataFrame,
    station: str,
    source: str,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Resolve raw duplicate timestamps without relying on source row order."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Duplicate resolution requires a DatetimeIndex.")

    duplicate_mask = df.index.duplicated(keep=False)
    duplicate_rows = df.loc[duplicate_mask]
    summary = {
        "Station": station_key(station),
        "Source": source,
        "Duplicate Timestamp Groups": 0,
        "Duplicate Input Rows": int(len(duplicate_rows)),
        "Rows Removed": 0,
        "Exact Duplicate Groups": 0,
        "Complementary Groups": 0,
        "Measurement Conflict Groups": 0,
        "Flag-Only Conflict Groups": 0,
    }
    if duplicate_rows.empty:
        return df.sort_index(kind="stable"), summary, []

    resolved_rows = []
    conflict_rows = []
    for timestamp, group in duplicate_rows.groupby(level=0, sort=True):
        summary["Duplicate Timestamp Groups"] += 1
        summary["Rows Removed"] += len(group) - 1
        values = group.reset_index(drop=True)
        conflicting = [
            column
            for column in values.columns
            if values[column].dropna().nunique() > 1
        ]

        if len(values.drop_duplicates()) == 1:
            conflict_type = "exact_duplicate"
            summary["Exact Duplicate Groups"] += 1
        elif not conflicting:
            conflict_type = "complementary"
            summary["Complementary Groups"] += 1
        elif set(conflicting) == {"Flag"}:
            conflict_type = "flag_only_conflict"
            summary["Flag-Only Conflict Groups"] += 1
        else:
            conflict_type = "measurement_conflict"
            summary["Measurement Conflict Groups"] += 1

        resolved = {}
        for column in values.columns:
            nonmissing = values[column].dropna().drop_duplicates()
            resolved[column] = nonmissing.iloc[0] if len(nonmissing) == 1 else np.nan
        resolved_rows.append(pd.Series(resolved, name=timestamp))

        if conflicting:
            source_rows = [
                {column: _json_value(value) for column, value in row.items()}
                for row in values.to_dict(orient="records")
            ]
            conflict_rows.append(
                {
                    "Station": station_key(station),
                    "Source": source,
                    "Timestamp": timestamp,
                    "Conflict Type": conflict_type,
                    "Row Count": int(len(values)),
                    "Conflicting Parameters": ";".join(conflicting),
                    "Source Rows": json.dumps(
                        source_rows,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

    nonduplicates = df.loc[~duplicate_mask]
    resolved_duplicates = pd.DataFrame(resolved_rows)
    resolved_duplicates.index = pd.DatetimeIndex(resolved_duplicates.index, name=df.index.name)
    result = pd.concat([nonduplicates, resolved_duplicates]).sort_index(kind="stable")
    if result.index.has_duplicates:
        raise ValueError("Duplicate resolution failed to produce a unique timestamp index.")
    return result, summary, conflict_rows


def record_duplicate_audit(
    df: pd.DataFrame,
    station: str,
    source: str,
    audit: DuplicateAudit | None,
) -> pd.DataFrame:
    resolved, summary, conflicts = resolve_duplicate_timestamps(df, station, source)
    if audit is not None:
        audit.summaries.append(summary)
        audit.conflicts.extend(conflicts)
        audit.source_coverage[source] = {
            "raw_rows": len(df), "resolved_timestamps": len(resolved),
            "start": str(resolved.index.min()), "end": str(resolved.index.max()),
            "invalid_ppt_before_hourly": int((resolved.Ppt.notna() &
                (~np.isfinite(resolved.Ppt) | resolved.Ppt.lt(0))).sum()) if "Ppt" in resolved else 0,
        }
    return resolved


def repair_concatenated_toa5_records(text):
    # A few TOA5 files contain two records glued together with no newline.
    return re.sub(r'(?<=[0-9])"(?=\d{4}-\d{2}-\d{2} )', '\n"', text)


def load_soil_data(station_id, base_dir, duplicate_audit=None):
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
    df = record_duplicate_audit(df, station_id, "soil", duplicate_audit)
    return aggregate_observations_to_hourly(df)


def load_met_data(station_id, base_dir, duplicate_audit=None):
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
        df = record_duplicate_audit(df, station_id, "met", duplicate_audit)
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
    df = record_duplicate_audit(df, station_id, "met", duplicate_audit)
    return aggregate_observations_to_hourly(df)


def merge_raw_data(station_id, soil_base_dir, met_base_dir, duplicate_audit=None):
    """Merge the union of source coverage with valid MET-first precipitation."""
    df_soil = load_soil_data(station_id, soil_base_dir, duplicate_audit)
    df_met = load_met_data(station_id, met_base_dir, duplicate_audit)
    return merge_hourly_sources(df_soil, df_met)


def merge_hourly_sources(df_soil, df_met):
    """Keep all source hours, including MET-only hours outside Soil coverage."""
    for source, frame in [("soil", df_soil), ("met", df_met)]:
        if not frame.empty:
            require_unique_datetime_index(frame, f"hourly {source} merge input")
            if not frame.index.equals(frame.index.floor("h")):
                raise ValueError(f"Hourly {source} merge input contains sub-hourly timestamps.")
    if df_met.empty:
        merged = df_soil.copy()
    elif df_soil.empty:
        merged = df_met.copy()
    else:
        merged = df_soil.join(df_met, how="outer", lsuffix="_soil", rsuffix="_met")
    for col in ["Ppt", "Ppt_soil", "Ppt_met"]:
        if col in merged:
            merged[col] = merged[col].where(np.isfinite(merged[col]) & merged[col].ge(0))
    if "Ppt_soil" in merged.columns and "Ppt_met" in merged.columns:
        merged["Ppt"] = merged["Ppt_met"].combine_first(merged["Ppt_soil"])
        merged.drop(columns=["Ppt_soil", "Ppt_met"], inplace=True)

    return complete_hourly_timeline(merged)


def complete_hourly_timeline(df):
    require_unique_datetime_index(df, "Stage 0 merged data")
    if df.empty:
        raise ValueError("Stage 0 has no valid source timestamps.")
    if not df.index.equals(df.index.floor("h")):
        raise ValueError("Stage 0 merged data must already be hourly.")
    timeline = pd.date_range(df.index.min(), df.index.max(), freq="h", name="Date")
    return df.reindex(timeline)


def write_duplicate_reports(audit: DuplicateAudit, station_id, output_dir) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    station = station_key(station_id)
    summary_path = output_dir / f"Station{station}_duplicate_summary.csv"
    conflict_path = output_dir / f"Station{station}_duplicate_conflicts.csv"
    pd.DataFrame(audit.summaries, columns=DUPLICATE_SUMMARY_COLUMNS).to_csv(
        summary_path,
        index=False,
    )
    pd.DataFrame(audit.conflicts, columns=DUPLICATE_CONFLICT_COLUMNS).to_csv(
        conflict_path,
        index=False,
    )
    return summary_path, conflict_path


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
    """Merge timestamp lists and de-duplicate invalid values already set to NaN."""
    combined = {}
    for key in set(missing) | set(wrong):
        combined[key] = sorted(set(missing.get(key, [])) | set(wrong.get(key, [])))
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
    full_df = complete_hourly_timeline(merged_df)
    return find_and_replace_wrong_data(full_df)


def file_fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(
        description="Generate merged, missing/invalid summary, and cleaned full-timeline CSVs for a soil station."
    )
    parser.add_argument("--station", "-s", type=str, default="1", help="Station ID or site code, e.g. 1 or CB01.")
    parser.add_argument("--soil-base-dir", type=str, default=str(DEFAULT_TXSON_DATA_DIR), help="Path to soil .dat files.")
    parser.add_argument("--met-base-dir", type=str, default=str(DEFAULT_TXSON_DATA_DIR), help="Path to MET .dat files.")
    parser.add_argument("--raw-output-dir", type=str, default="raw_merged_data", help="Directory for merged CSVs.")
    parser.add_argument(
        "--duplicate-report-dir",
        type=str,
        default="duplicate_resolution_reports",
        help="Directory for per-station duplicate-resolution reports.",
    )
    parser.add_argument("--missing-output", type=str, default=None, help="Filename for missing/invalid summary CSV.")
    parser.add_argument("--cleaned-output", type=str, default=None, help="Filename for cleaned full-timeline CSV.")
    parser.add_argument("--provenance-dir", default="stage0_reports", help="Directory for per-station source/code/output manifests.")
    args = parser.parse_args()

    station_id = args.station

    duplicate_audit = DuplicateAudit()
    merged_df = merge_raw_data(
        station_id,
        args.soil_base_dir,
        args.met_base_dir,
        duplicate_audit,
    )
    duplicate_summary, duplicate_conflicts = write_duplicate_reports(
        duplicate_audit,
        station_id,
        args.duplicate_report_dir,
    )
    print(f"Duplicate summary saved to: {duplicate_summary}")
    print(f"Duplicate conflicts saved to: {duplicate_conflicts}")
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

    sources = [resolve_station_file(args.soil_base_dir, station_id, ["SM_{station}.dat", "{station}.dat"]),
               resolve_station_file(args.met_base_dir, station_id, ["MET_{station}.dat", "{station}_met.dat"], required=False)]
    manifest = {
        "station": station_key(station_id),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_policy": "complete hourly union of Soil and MET source timestamps",
        "ppt_policy": "finite nonnegative MET first; finite nonnegative Soil fallback; otherwise NaN",
        "duplicate_policy": "exact collapse; complementary merge; conflicting cells missing and reported",
        "code": [file_fingerprint(__file__), file_fingerprint(Path(__file__).with_name("time_index_utils.py"))],
        "sources": [file_fingerprint(path) for path in sources if path is not None],
        "source_coverage": duplicate_audit.source_coverage,
        "outputs": [file_fingerprint(path) for path in [raw_path, clean_out, miss_out, duplicate_summary, duplicate_conflicts]],
        "rows": len(cleaned_df), "start": str(cleaned_df.index.min()), "end": str(cleaned_df.index.max()),
        "missing_by_parameter": {col: int(cleaned_df[col].isna().sum()) for col in cleaned_df},
        "range_rejected_by_parameter": {col: len(times) for col, times in wrong.items()},
    }
    provenance_dir = Path(args.provenance_dir)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / f"Station{station_id}_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

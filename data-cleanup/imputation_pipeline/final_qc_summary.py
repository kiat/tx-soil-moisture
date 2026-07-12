"""Build final QC summaries after staged soil-gap filling.

This script does not modify station data. It audits the latest repaired output
files and writes CSV summaries for the remaining issues that need a final
decision: residual NaN gaps, unavailable sensor columns, exact bound values,
near-zero/flat sensors, and very-long-gap segments marked for review.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from param_config import ALL_SOIL_PARAMS


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "final_qc_reports"

SWC_PARAMS = [p for p in ALL_SOIL_PARAMS if p.startswith("SWC_")]
TEMP_PARAMS = [p for p in ALL_SOIL_PARAMS if p.startswith("T_")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize final QC issues after gap filling.")
    parser.add_argument("--station", type=str, nargs="*", help="Station IDs/site codes to audit.")
    parser.add_argument("--param", type=str, nargs="*", help="Parameters to audit.")
    parser.add_argument("--swc-near-zero", type=float, default=0.01)
    parser.add_argument("--swc-flat-range", type=float, default=0.01)
    parser.add_argument("--temp-flat-range", type=float, default=1.0)
    parser.add_argument("--dominant-fraction", type=float, default=0.5)
    parser.add_argument("--min-sensor-hours", type=int, default=720)
    return parser.parse_args()


def latest_path_for(station: str) -> Path:
    candidates = [
        OUT_DIR / f"Station{station}_filled_final.csv",
        OUT_DIR / f"Station{station}_filled_manual_qc.csv",
        OUT_DIR / f"Station{station}_filled_sensor_qc.csv",
        OUT_DIR / f"Station{station}_filled_verylonggaps_repaired.csv",
        OUT_DIR / f"Station{station}_filled_verylonggaps.csv",
        OUT_DIR / f"Station{station}_filled_longgaps_repaired.csv",
        OUT_DIR / f"Station{station}_filled_longgaps.csv",
        OUT_DIR / f"Station{station}_filled_mediumgaps_repaired.csv",
        OUT_DIR / f"Station{station}_filled_mediumgaps.csv",
        OUT_DIR / f"Station{station}_filled_shortgaps.csv",
        BASE_DIR / "cleaned_data" / f"Station{station}_cleaned_data.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def discover_stations() -> List[str]:
    pat = re.compile(r"Station(.+)_filled_verylonggaps_repaired\.csv")
    stations = [
        m.group(1)
        for path in OUT_DIR.glob("Station*_filled_verylonggaps_repaired.csv")
        if (m := pat.match(path.name))
    ]
    if stations:
        return sorted(stations)

    fallback_pat = re.compile(r"Station(.+)_filled_verylonggaps\.csv")
    return sorted(
        m.group(1)
        for path in OUT_DIR.glob("Station*_filled_verylonggaps.csv")
        if (m := fallback_pat.match(path.name))
    )


def read_station(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    return df.sort_index()


def gap_category(hours: int) -> str:
    if hours < 24:
        return "short_<24h"
    if hours < 168:
        return "medium_24-167h"
    if hours < 720:
        return "long_168-719h"
    return "verylong_>=720h"


def nan_runs(df: pd.DataFrame, station: str, param: str) -> List[dict]:
    if param not in df.columns:
        return []
    missing = df[param].isna().to_numpy()
    rows: List[dict] = []
    i = 0
    while i < len(missing):
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < len(missing) and missing[j]:
            j += 1
        hours = j - i
        rows.append(
            {
                "Station": station,
                "Parameter": param,
                "Start": df.index[i],
                "End": df.index[j - 1],
                "Hours": hours,
                "Category": gap_category(hours),
            }
        )
        i = j
    return rows


def longest_equal_run(series: pd.Series) -> Tuple[int, object]:
    s = series.dropna()
    if s.empty:
        return 0, pd.NA
    groups = s.ne(s.shift()).cumsum()
    counts = s.groupby(groups).size()
    idx = counts.idxmax()
    value = s[groups == idx].iloc[0]
    return int(counts.loc[idx]), value


def sensor_flags(
    station: str,
    param: str,
    series: pd.Series,
    args: argparse.Namespace,
) -> Tuple[List[str], dict]:
    s = pd.to_numeric(series, errors="coerce")
    non_missing = s.dropna()
    flags: List[str] = []

    if param.startswith("SWC_"):
        lower, upper = 0.0, 0.6
        exact_lower = int((non_missing == lower).sum())
        exact_upper = int((non_missing == upper).sum())
        near_zero = int((non_missing <= args.swc_near_zero).sum())
        near_zero_fraction = near_zero / len(non_missing) if len(non_missing) else 0.0
        data_range = float(non_missing.max() - non_missing.min()) if len(non_missing) else pd.NA
        if len(non_missing) >= args.min_sensor_hours and near_zero_fraction >= args.dominant_fraction:
            flags.append("swc_near_zero_dominant")
        if len(non_missing) >= args.min_sensor_hours and pd.notna(data_range) and data_range <= args.swc_flat_range:
            flags.append("swc_low_variability")
    else:
        lower, upper = -30.0, 60.0
        exact_lower = int((non_missing == lower).sum())
        exact_upper = int((non_missing == upper).sum())
        near_zero = 0
        near_zero_fraction = 0.0
        data_range = float(non_missing.max() - non_missing.min()) if len(non_missing) else pd.NA
        if len(non_missing) >= args.min_sensor_hours and pd.notna(data_range) and data_range <= args.temp_flat_range:
            flags.append("temperature_low_variability")

    if exact_lower:
        flags.append("exact_lower_bound_values")
    if exact_upper:
        flags.append("exact_upper_bound_values")

    longest_run, longest_value = longest_equal_run(s)
    if longest_run >= args.min_sensor_hours:
        flags.append("long_constant_run")

    metrics = {
        "Nonmissing Hours": int(len(non_missing)),
        "NaN Hours": int(s.isna().sum()),
        "NaN Fraction": float(s.isna().mean()) if len(s) else pd.NA,
        "Min": float(non_missing.min()) if len(non_missing) else pd.NA,
        "Max": float(non_missing.max()) if len(non_missing) else pd.NA,
        "Mean": float(non_missing.mean()) if len(non_missing) else pd.NA,
        "Std": float(non_missing.std()) if len(non_missing) else pd.NA,
        "Range": data_range,
        "Exact Lower Bound Count": exact_lower,
        "Exact Upper Bound Count": exact_upper,
        "SWC Near-Zero Count": near_zero,
        "SWC Near-Zero Fraction": near_zero_fraction,
        "Longest Constant Run Hours": longest_run,
        "Longest Constant Run Value": longest_value,
    }
    return sorted(set(flags)), metrics


def summarize_verylong_review() -> pd.DataFrame:
    path = BASE_DIR / "verylonggaps_review_segments.csv"
    if not path.exists():
        return pd.DataFrame()
    review = pd.read_csv(path)
    if review.empty:
        return pd.DataFrame()
    return (
        review.groupby(["Station", "Parameter", "Status", "Review Reason"], dropna=False)
        .agg(
            Segments=("Status", "size"),
            Filled_Hours=("Filled Hours", "sum"),
            Repaired_Points=("Repaired Points", "sum"),
            Max_Hourly_Change=("Max Hourly Change", "max"),
            Max_Donor_Mean_Fraction=("Donor Mean Fraction", "max"),
        )
        .reset_index()
        .sort_values(["Repaired_Points", "Filled_Hours"], ascending=False)
    )


def write_outputs(
    overview_rows: List[dict],
    param_rows: List[dict],
    gap_rows: List[dict],
    missing_column_rows: List[dict],
    suspicious_rows: List[dict],
) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    overview = pd.DataFrame(overview_rows)
    param_summary = pd.DataFrame(param_rows)
    remaining_gaps = pd.DataFrame(gap_rows)
    missing_columns = pd.DataFrame(missing_column_rows)
    suspicious = pd.DataFrame(suspicious_rows)
    verylong_review = summarize_verylong_review()

    overview.to_csv(REPORT_DIR / "final_qc_overview.csv", index=False)
    param_summary.to_csv(REPORT_DIR / "final_qc_station_parameter_summary.csv", index=False)
    remaining_gaps.to_csv(REPORT_DIR / "final_qc_remaining_nan_runs.csv", index=False)
    missing_columns.to_csv(REPORT_DIR / "final_qc_missing_sensor_columns.csv", index=False)
    suspicious.to_csv(REPORT_DIR / "final_qc_suspicious_sensors.csv", index=False)
    verylong_review.to_csv(REPORT_DIR / "final_qc_verylong_review_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    stations = args.station if args.station else discover_stations()
    params = args.param if args.param else ALL_SOIL_PARAMS

    overview_rows: List[dict] = []
    param_rows: List[dict] = []
    gap_rows: List[dict] = []
    missing_column_rows: List[dict] = []
    suspicious_rows: List[dict] = []

    for station in stations:
        path = latest_path_for(station)
        if not path.exists():
            overview_rows.append({"Station": station, "Input File": str(path), "Status": "missing_input"})
            continue
        df = read_station(path)
        station_nan = 0

        for param in params:
            if param not in df.columns:
                missing_column_rows.append({"Station": station, "Parameter": param, "Reason": "column_missing"})
                continue

            runs = nan_runs(df, station, param)
            gap_rows.extend(runs)
            nan_hours = sum(int(r["Hours"]) for r in runs)
            station_nan += nan_hours
            run_counts = pd.Series([r["Category"] for r in runs]).value_counts().to_dict() if runs else {}

            flags, metrics = sensor_flags(station, param, df[param], args)
            summary = {
                "Station": station,
                "Parameter": param,
                "Input File": path.name,
                "Start": df.index.min(),
                "End": df.index.max(),
                "Total Hours": int(len(df)),
                "NaN Runs": int(len(runs)),
                "Max NaN Run Hours": max([int(r["Hours"]) for r in runs], default=0),
                "Short NaN Runs": int(run_counts.get("short_<24h", 0)),
                "Medium NaN Runs": int(run_counts.get("medium_24-167h", 0)),
                "Long NaN Runs": int(run_counts.get("long_168-719h", 0)),
                "VeryLong NaN Runs": int(run_counts.get("verylong_>=720h", 0)),
                "Flags": ";".join(flags),
            }
            summary.update(metrics)
            param_rows.append(summary)

            if flags:
                suspicious_rows.append(summary)

        overview_rows.append(
            {
                "Station": station,
                "Input File": path.name,
                "Status": "ok",
                "Rows": int(len(df)),
                "Remaining Soil NaN Hours": int(station_nan),
                "Missing Soil Columns": ";".join(
                    sorted(row["Parameter"] for row in missing_column_rows if row["Station"] == station)
                ),
            }
        )

    write_outputs(overview_rows, param_rows, gap_rows, missing_column_rows, suspicious_rows)

    remaining = pd.DataFrame(gap_rows)
    suspicious = pd.DataFrame(suspicious_rows)
    missing_columns = pd.DataFrame(missing_column_rows)

    print("Final QC summary complete.")
    print(f"Stations audited: {len(stations)}")
    print(f"Remaining NaN runs: {len(remaining)}")
    print(f"Remaining NaN hours: {int(remaining['Hours'].sum()) if not remaining.empty else 0}")
    print(f"Missing sensor columns: {len(missing_columns)}")
    print(f"Suspicious station/parameter rows: {len(suspicious)}")
    print(f"Outputs written under: {REPORT_DIR}")


if __name__ == "__main__":
    main()

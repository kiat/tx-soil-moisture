"""Validate and repair long-gap outputs.

This script audits XGBoost long-gap fills after Longgaps.py has run. It keeps
accepted fills, restores suspicious filled segments to NaN, and writes repaired
outputs separately so the raw long-gap run remains available.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from param_config import ALL_SOIL_PARAMS


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
MISS_DIR = BASE_DIR / "missing_data"

SWC_BOUNDS = (0.0, 0.6)
TEMP_BOUNDS = (-30.0, 60.0)


@dataclass(frozen=True)
class SegmentKey:
    station: str
    parameter: str
    start: pd.Timestamp
    end: pd.Timestamp


def discover_stations() -> List[str]:
    pat = re.compile(r"Station(.+)_filled_longgaps\.csv")
    return sorted(
        m.group(1)
        for fn in OUT_DIR.glob("Station*_filled_longgaps.csv")
        if (m := pat.match(fn.name))
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Validate and repair long-gap fills.")
    p.add_argument("--station", type=str, nargs="*", help="Station IDs/site codes to validate.")
    p.add_argument("--param", type=str, nargs="*", help="Parameters to validate.")
    p.add_argument(
        "--write-repaired",
        action="store_true",
        help="Write Station*_filled_longgaps_repaired.csv files.",
    )
    p.add_argument("--max-swc-jump", type=float, default=0.08)
    p.add_argument("--max-temp-jump", type=float, default=8.0)
    p.add_argument("--max-swc-boundary-jump", type=float, default=0.05)
    p.add_argument("--max-temp-boundary-jump", type=float, default=5.0)
    p.add_argument("--temp-context-margin", type=float, default=12.0)
    p.add_argument("--context-days", type=int, default=14)
    return p.parse_args()


def read_series_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def source_path_for(station: str) -> Path:
    candidates = [
        OUT_DIR / f"Station{station}_filled_mediumgaps_repaired.csv",
        OUT_DIR / f"Station{station}_filled_mediumgaps.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def load_expected_segments(stations: Iterable[str], params: Iterable[str]) -> Dict[SegmentKey, int]:
    selected_stations = set(stations)
    selected_params = set(params)
    expected: Dict[SegmentKey, int] = {}

    for miss_path in sorted(MISS_DIR.glob("Station*_missing_data.csv")):
        station = miss_path.name[len("Station") : -len("_missing_data.csv")]
        if station not in selected_stations:
            continue

        miss = pd.read_csv(miss_path, parse_dates=["Start Timestamp", "End Timestamp"])
        if miss.empty or "Number Missing" not in miss.columns:
            continue
        miss["Number Missing"] = pd.to_numeric(miss["Number Missing"], errors="coerce")
        mask = (
            miss["Parameter"].isin(selected_params)
            & miss["Number Missing"].between(168, 720, inclusive="both")
        )
        for _, row in miss.loc[mask].iterrows():
            expected[
                SegmentKey(
                    station=station,
                    parameter=row["Parameter"],
                    start=row["Start Timestamp"],
                    end=row["End Timestamp"],
                )
            ] = int(row["Number Missing"])
    return expected


def load_filled_segments(stations: Iterable[str], params: Iterable[str]) -> Dict[SegmentKey, pd.DataFrame]:
    selected_stations = set(stations)
    selected_params = set(params)
    filled: Dict[SegmentKey, pd.DataFrame] = {}

    for detail_path in sorted(OUT_DIR.glob("Station*_longgap_fill_detail.csv")):
        detail = pd.read_csv(detail_path, parse_dates=["Start", "End", "Timestamp"])
        if detail.empty or "Station" not in detail.columns:
            continue
        detail["Station"] = detail["Station"].astype(str)
        detail = detail[
            detail["Station"].isin(selected_stations)
            & detail["Parameter"].isin(selected_params)
        ]
        for (station, param, start, end), group in detail.groupby(
            ["Station", "Parameter", "Start", "End"], sort=True
        ):
            filled[SegmentKey(str(station), param, start, end)] = group.sort_values("Timestamp").copy()
    return filled


def context_values(
    source: pd.DataFrame,
    param: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    days: int,
) -> pd.Series:
    before = source.loc[start - pd.Timedelta(days=days) : start - pd.Timedelta(hours=1), param]
    after = source.loc[end + pd.Timedelta(hours=1) : end + pd.Timedelta(days=days), param]
    return pd.concat([before, after]).dropna()


def segment_metrics(
    key: SegmentKey,
    group: pd.DataFrame,
    long_df: pd.DataFrame,
    source_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[str, str, Dict[str, object]]:
    values = group["Filled"].astype(float)
    timestamps = pd.DatetimeIndex(group["Timestamp"])
    expected_index = pd.date_range(key.start, key.end, freq="h")
    reasons: List[str] = []

    if key.parameter.startswith("SWC_"):
        lower, upper = SWC_BOUNDS
        jump_limit = args.max_swc_jump
        boundary_limit = args.max_swc_boundary_jump
    else:
        lower, upper = TEMP_BOUNDS
        jump_limit = args.max_temp_jump
        boundary_limit = args.max_temp_boundary_jump

    lower_hits = int((values <= lower).sum())
    upper_hits = int((values >= upper).sum())
    if lower_hits or upper_hits:
        reasons.append("hit_physical_bound")

    if len(timestamps) != len(expected_index) or not timestamps.equals(expected_index):
        reasons.append("timestamp_mismatch")
    if values.isna().any():
        reasons.append("filled_value_nan")

    max_hourly_change = float("nan")
    if key.parameter in long_df.columns:
        x = long_df.loc[key.start : key.end, key.parameter]
        diffs = x.diff().abs().dropna()
        if len(diffs):
            max_hourly_change = float(diffs.max())
            if max_hourly_change > jump_limit:
                reasons.append("large_hourly_jump")

    start_jump = end_jump = float("nan")
    if key.parameter in source_df.columns:
        left_ts = key.start - pd.Timedelta(hours=1)
        right_ts = key.end + pd.Timedelta(hours=1)
        left = source_df.loc[left_ts, key.parameter] if left_ts in source_df.index else pd.NA
        right = source_df.loc[right_ts, key.parameter] if right_ts in source_df.index else pd.NA
        first = values.iloc[0] if len(values) else pd.NA
        last = values.iloc[-1] if len(values) else pd.NA
        if pd.notna(left) and pd.notna(first):
            start_jump = float(abs(first - left))
            if start_jump > boundary_limit:
                reasons.append("large_start_boundary_jump")
        if pd.notna(right) and pd.notna(last):
            end_jump = float(abs(right - last))
            if end_jump > boundary_limit:
                reasons.append("large_end_boundary_jump")

        if key.parameter.startswith("T_"):
            ctx = context_values(source_df, key.parameter, key.start, key.end, args.context_days)
            if len(ctx) >= 24:
                context_min = float(ctx.min())
                context_max = float(ctx.max())
                if float(values.min()) < context_min - args.temp_context_margin:
                    reasons.append("below_local_temperature_context")
                if float(values.max()) > context_max + args.temp_context_margin:
                    reasons.append("above_local_temperature_context")
            else:
                context_min = context_max = float("nan")
        else:
            context_min = context_max = float("nan")
    else:
        context_min = context_max = float("nan")

    status = "rejected" if reasons else "accepted"
    metrics = {
        "Filled Hours": int(len(values)),
        "Filled Min": float(values.min()) if len(values) else pd.NA,
        "Filled Max": float(values.max()) if len(values) else pd.NA,
        "Lower Bound Hits": lower_hits,
        "Upper Bound Hits": upper_hits,
        "Max Hourly Change": max_hourly_change,
        "Start Boundary Jump": start_jump,
        "End Boundary Jump": end_jump,
        "Context Min": context_min,
        "Context Max": context_max,
    }
    return status, ";".join(sorted(set(reasons))), metrics


def build_summary(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[SegmentKey, pd.DataFrame]]:
    stations = args.station if args.station else discover_stations()
    params = args.param if args.param else ALL_SOIL_PARAMS
    expected = load_expected_segments(stations, params)
    filled = load_filled_segments(stations, params)

    all_keys = sorted(set(expected) | set(filled), key=lambda k: (k.station, k.parameter, k.start, k.end))
    long_cache: Dict[str, pd.DataFrame] = {}
    source_cache: Dict[str, pd.DataFrame] = {}
    accepted_details: Dict[SegmentKey, pd.DataFrame] = {}
    rows: List[Dict[str, object]] = []

    for key in all_keys:
        row = {
            "Station": key.station,
            "Parameter": key.parameter,
            "Start": key.start,
            "End": key.end,
            "Expected Hours": expected.get(key, pd.NA),
        }

        if key in filled:
            if key.station not in long_cache:
                long_cache[key.station] = read_series_file(OUT_DIR / f"Station{key.station}_filled_longgaps.csv")
            if key.station not in source_cache:
                source_cache[key.station] = read_series_file(source_path_for(key.station))
            status, reason, metrics = segment_metrics(
                key, filled[key], long_cache[key.station], source_cache[key.station], args
            )
            row.update(metrics)
            row["Status"] = status
            row["Reason"] = reason
            if status == "accepted":
                accepted_details[key] = filled[key]
        else:
            row.update(
                {
                    "Status": "skipped",
                    "Reason": "not_in_fill_detail",
                    "Filled Hours": 0,
                }
            )
        rows.append(row)

    return pd.DataFrame(rows), accepted_details


def write_repaired_outputs(
    summary: pd.DataFrame,
    accepted_details: Dict[SegmentKey, pd.DataFrame],
    stations: Iterable[str],
) -> None:
    rejected = summary[summary["Status"] == "rejected"]
    rejected_keys = {
        SegmentKey(row.Station, row.Parameter, row.Start, row.End)
        for row in rejected.itertuples(index=False)
    }

    accepted_by_station: Dict[str, List[pd.DataFrame]] = {}
    for key, detail in accepted_details.items():
        accepted_by_station.setdefault(key.station, []).append(detail)

    for station in stations:
        long_path = OUT_DIR / f"Station{station}_filled_longgaps.csv"
        if not long_path.exists():
            continue
        repaired = read_series_file(long_path)
        for key in rejected_keys:
            if key.station != station or key.parameter not in repaired.columns:
                continue
            idx = pd.date_range(key.start, key.end, freq="h")
            repaired.loc[repaired.index.intersection(idx), key.parameter] = pd.NA

        repaired.to_csv(OUT_DIR / f"Station{station}_filled_longgaps_repaired.csv")

        details = accepted_by_station.get(station, [])
        if details:
            pd.concat(details, ignore_index=True).to_csv(
                OUT_DIR / f"Station{station}_longgap_fill_detail_repaired.csv",
                index=False,
            )


def write_summaries(summary: pd.DataFrame) -> None:
    summary_path = BASE_DIR / "longgaps_validation_summary.csv"
    rejected_path = BASE_DIR / "longgaps_rejected_segments.csv"
    station_path = BASE_DIR / "longgaps_validation_station_summary.csv"

    summary.to_csv(summary_path, index=False)
    summary[summary["Status"] == "rejected"].to_csv(rejected_path, index=False)

    station_summary = (
        summary.groupby(["Station", "Status"], dropna=False)
        .agg(
            Segments=("Status", "size"),
            Expected_Hours=("Expected Hours", "sum"),
            Filled_Hours=("Filled Hours", "sum"),
        )
        .reset_index()
    )
    station_summary.to_csv(station_path, index=False)


def main() -> None:
    args = parse_args()
    stations = args.station if args.station else discover_stations()
    summary, accepted_details = build_summary(args)
    write_summaries(summary)

    if args.write_repaired:
        write_repaired_outputs(summary, accepted_details, stations)

    counts = summary["Status"].value_counts().to_dict()
    print("Long-gap validation complete.")
    print("Segments:", counts)
    print(f"Summary: {BASE_DIR / 'longgaps_validation_summary.csv'}")
    print(f"Rejected: {BASE_DIR / 'longgaps_rejected_segments.csv'}")
    print(f"Station summary: {BASE_DIR / 'longgaps_validation_station_summary.csv'}")
    if args.write_repaired:
        print("Repaired station files written to output/*_filled_longgaps_repaired.csv")
    else:
        print("Dry run only. Add --write-repaired to write repaired station files.")


if __name__ == "__main__":
    main()

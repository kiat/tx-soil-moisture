"""Validate and repair very-long-gap outputs.

Very-long gaps can span months or years, so this validator is intentionally
less destructive than the medium/long validators. It writes segment-level
validation reports, and when repair is requested it restores only clearly
suspicious filled timestamps to NaN instead of rejecting the entire segment.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Validate and repair very-long-gap fills.")
    p.add_argument("--station", type=str, nargs="*", help="Station IDs/site codes to validate.")
    p.add_argument("--param", type=str, nargs="*", help="Parameters to validate.")
    p.add_argument(
        "--write-repaired",
        action="store_true",
        help="Write Station*_filled_verylonggaps_repaired.csv files.",
    )
    p.add_argument("--max-swc-jump-review", type=float, default=0.12)
    p.add_argument("--max-swc-jump-repair", type=float, default=0.20)
    p.add_argument("--max-temp-jump-review", type=float, default=8.0)
    p.add_argument("--max-temp-jump-repair", type=float, default=12.0)
    p.add_argument("--max-swc-boundary-jump", type=float, default=0.08)
    p.add_argument("--max-temp-boundary-jump", type=float, default=8.0)
    p.add_argument("--donor-mean-review-fraction", type=float, default=0.5)
    return p.parse_args()


def discover_stations() -> List[str]:
    pat = re.compile(r"Station(.+)_filled_verylonggaps\.csv")
    return sorted(
        m.group(1)
        for fn in OUT_DIR.glob("Station*_filled_verylonggaps.csv")
        if (m := pat.match(fn.name))
    )


def read_series_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    return df.sort_index()


def source_path_for(station: str) -> Path:
    candidates = [
        OUT_DIR / f"Station{station}_filled_longgaps_repaired.csv",
        OUT_DIR / f"Station{station}_filled_longgaps.csv",
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
        mask = miss["Parameter"].isin(selected_params) & (miss["Number Missing"] >= 720)
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

    for detail_path in sorted(OUT_DIR.glob("Station*_verylonggap_fill_detail.csv")):
        detail = pd.read_csv(detail_path, parse_dates=["Start", "End", "Timestamp"], low_memory=False)
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


def parameter_limits(param: str, args: argparse.Namespace) -> Tuple[float, float, float, float, float]:
    if param.startswith("SWC_"):
        lower, upper = SWC_BOUNDS
        return lower, upper, args.max_swc_jump_review, args.max_swc_jump_repair, args.max_swc_boundary_jump
    lower, upper = TEMP_BOUNDS
    return lower, upper, args.max_temp_jump_review, args.max_temp_jump_repair, args.max_temp_boundary_jump


def segment_metrics(
    key: SegmentKey,
    group: pd.DataFrame,
    verylong_df: pd.DataFrame,
    source_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Set[pd.Timestamp]]:
    values = group["Filled"].astype(float)
    timestamps = pd.DatetimeIndex(group["Timestamp"])
    expected_index = pd.date_range(key.start, key.end, freq="h")
    lower, upper, jump_review, jump_repair, boundary_limit = parameter_limits(key.parameter, args)

    reasons: List[str] = []
    review_reasons: List[str] = []
    repair_points: Set[pd.Timestamp] = set()

    if len(timestamps) != len(expected_index) or not timestamps.equals(expected_index):
        reasons.append("timestamp_mismatch")
    if values.isna().any():
        reasons.append("filled_value_nan")

    lower_hits = int((values <= lower).sum())
    upper_hits = int((values >= upper).sum())
    if lower_hits or upper_hits:
        review_reasons.append("hit_or_clipped_physical_bound")
        repair_points.update(timestamps[(values <= lower) | (values >= upper)])

    max_hourly_change = float("nan")
    severe_jump_count = 0
    review_jump_count = 0
    if key.parameter in verylong_df.columns:
        segment = verylong_df.loc[key.start : key.end, key.parameter]
        diffs = segment.diff().abs().dropna()
        if len(diffs):
            max_hourly_change = float(diffs.max())
            review_jump_count = int((diffs > jump_review).sum())
            severe = diffs[diffs > jump_repair]
            severe_jump_count = int(len(severe))
            if review_jump_count:
                review_reasons.append("large_hourly_jump")
            repair_points.update(pd.DatetimeIndex(severe.index))

    start_jump = end_jump = float("nan")
    if key.parameter in source_df.columns and len(values):
        left_ts = key.start - pd.Timedelta(hours=1)
        right_ts = key.end + pd.Timedelta(hours=1)
        left = source_df.loc[left_ts, key.parameter] if left_ts in source_df.index else pd.NA
        right = source_df.loc[right_ts, key.parameter] if right_ts in source_df.index else pd.NA
        first = values.iloc[0]
        last = values.iloc[-1]
        if pd.notna(left):
            start_jump = float(abs(first - left))
            if start_jump > boundary_limit:
                review_reasons.append("large_start_boundary_jump")
                repair_points.add(timestamps[0])
        if pd.notna(right):
            end_jump = float(abs(right - last))
            if end_jump > boundary_limit:
                review_reasons.append("large_end_boundary_jump")
                repair_points.add(timestamps[-1])

    method_counts = group["Method"].value_counts(dropna=False).to_dict() if "Method" in group.columns else {}
    donor_mean_hours = int(method_counts.get("donor_mean", 0))
    donor_mean_fraction = donor_mean_hours / len(group) if len(group) else 0.0
    if donor_mean_fraction >= args.donor_mean_review_fraction:
        review_reasons.append("high_donor_mean_fraction")

    corr = pd.to_numeric(group.get("Abs Corr", pd.Series(dtype=float)), errors="coerce").dropna()
    min_corr = float(corr.min()) if len(corr) else float("nan")
    median_corr = float(corr.median()) if len(corr) else float("nan")

    status = "accepted"
    if reasons:
        status = "rejected"
    elif repair_points:
        status = "repaired"
    elif review_reasons:
        status = "review"

    metrics = {
        "Status": status,
        "Reason": ";".join(sorted(set(reasons))),
        "Review Reason": ";".join(sorted(set(review_reasons))),
        "Filled Hours": int(len(values)),
        "Repaired Points": int(len(repair_points)),
        "Filled Min": float(values.min()) if len(values) else pd.NA,
        "Filled Max": float(values.max()) if len(values) else pd.NA,
        "Lower Bound Hits": lower_hits,
        "Upper Bound Hits": upper_hits,
        "Max Hourly Change": max_hourly_change,
        "Review Jump Count": review_jump_count,
        "Severe Jump Count": severe_jump_count,
        "Start Boundary Jump": start_jump,
        "End Boundary Jump": end_jump,
        "Donor Mean Hours": donor_mean_hours,
        "Donor Mean Fraction": donor_mean_fraction,
        "Min Abs Corr": min_corr,
        "Median Abs Corr": median_corr,
    }
    return metrics, repair_points


def build_summary(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[SegmentKey, Set[pd.Timestamp]]]:
    stations = args.station if args.station else discover_stations()
    params = args.param if args.param else ALL_SOIL_PARAMS
    expected = load_expected_segments(stations, params)
    filled = load_filled_segments(stations, params)
    all_keys = sorted(set(expected) | set(filled), key=lambda k: (k.station, k.parameter, k.start, k.end))

    verylong_cache: Dict[str, pd.DataFrame] = {}
    source_cache: Dict[str, pd.DataFrame] = {}
    repair_points: Dict[SegmentKey, Set[pd.Timestamp]] = {}
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
            if key.station not in verylong_cache:
                verylong_cache[key.station] = read_series_file(OUT_DIR / f"Station{key.station}_filled_verylonggaps.csv")
            if key.station not in source_cache:
                source_cache[key.station] = read_series_file(source_path_for(key.station))
            metrics, points = segment_metrics(
                key, filled[key], verylong_cache[key.station], source_cache[key.station], args
            )
            row.update(metrics)
            repair_points[key] = points
        else:
            row.update(
                {
                    "Status": "skipped",
                    "Reason": "not_in_fill_detail",
                    "Review Reason": "",
                    "Filled Hours": 0,
                    "Repaired Points": 0,
                }
            )
        rows.append(row)

    return pd.DataFrame(rows), repair_points


def write_repaired_outputs(
    summary: pd.DataFrame,
    repair_points: Dict[SegmentKey, Set[pd.Timestamp]],
    stations: Iterable[str],
) -> None:
    points_by_station: Dict[str, List[Tuple[str, pd.Timestamp]]] = {}
    for key, points in repair_points.items():
        for ts in points:
            points_by_station.setdefault(key.station, []).append((key.parameter, ts))

    for station in stations:
        verylong_path = OUT_DIR / f"Station{station}_filled_verylonggaps.csv"
        if not verylong_path.exists():
            continue
        repaired = read_series_file(verylong_path)
        for param, ts in points_by_station.get(station, []):
            if param in repaired.columns and ts in repaired.index:
                repaired.loc[ts, param] = pd.NA
        repaired.index.name = "Date"
        repaired.to_csv(OUT_DIR / f"Station{station}_filled_verylonggaps_repaired.csv", na_rep="NaN")

        detail_path = OUT_DIR / f"Station{station}_verylonggap_fill_detail.csv"
        if detail_path.exists():
            detail = pd.read_csv(detail_path, parse_dates=["Timestamp", "Start", "End"], low_memory=False)
            if not detail.empty:
                repaired_pairs = {(param, ts) for param, ts in points_by_station.get(station, [])}
                keep = [
                    (row.Parameter, row.Timestamp) not in repaired_pairs
                    for row in detail.itertuples(index=False)
                ]
                detail.loc[keep].to_csv(
                    OUT_DIR / f"Station{station}_verylonggap_fill_detail_repaired.csv",
                    index=False,
                )


def write_summaries(summary: pd.DataFrame, repair_points: Dict[SegmentKey, Set[pd.Timestamp]]) -> None:
    summary_path = BASE_DIR / "verylonggaps_validation_summary.csv"
    review_path = BASE_DIR / "verylonggaps_review_segments.csv"
    repaired_points_path = BASE_DIR / "verylonggaps_repaired_points.csv"
    station_path = BASE_DIR / "verylonggaps_validation_station_summary.csv"

    summary.to_csv(summary_path, index=False)
    summary[summary["Status"].isin(["review", "repaired", "rejected", "skipped"])].to_csv(review_path, index=False)

    point_rows: List[Dict[str, object]] = []
    for key, points in repair_points.items():
        for ts in sorted(points):
            point_rows.append(
                {
                    "Station": key.station,
                    "Parameter": key.parameter,
                    "Segment Start": key.start,
                    "Segment End": key.end,
                    "Timestamp": ts,
                }
            )
    pd.DataFrame(point_rows).to_csv(repaired_points_path, index=False)

    station_summary = (
        summary.groupby(["Station", "Status"], dropna=False)
        .agg(
            Segments=("Status", "size"),
            Expected_Hours=("Expected Hours", "sum"),
            Filled_Hours=("Filled Hours", "sum"),
            Repaired_Points=("Repaired Points", "sum"),
        )
        .reset_index()
    )
    station_summary.to_csv(station_path, index=False)


def main() -> None:
    args = parse_args()
    stations = args.station if args.station else discover_stations()
    summary, repair_points = build_summary(args)
    write_summaries(summary, repair_points)

    if args.write_repaired:
        write_repaired_outputs(summary, repair_points, stations)

    counts = summary["Status"].value_counts().to_dict()
    print("Very-long-gap validation complete.")
    print("Segments:", counts)
    print("Total repaired points:", int(summary["Repaired Points"].sum()))
    print(f"Summary: {BASE_DIR / 'verylonggaps_validation_summary.csv'}")
    print(f"Review/repaired segments: {BASE_DIR / 'verylonggaps_review_segments.csv'}")
    print(f"Repaired points: {BASE_DIR / 'verylonggaps_repaired_points.csv'}")
    print(f"Station summary: {BASE_DIR / 'verylonggaps_validation_station_summary.csv'}")
    if args.write_repaired:
        print("Repaired station files written to output/*_filled_verylonggaps_repaired.csv")
    else:
        print("Dry run only. Add --write-repaired to write repaired station files.")


if __name__ == "__main__":
    main()

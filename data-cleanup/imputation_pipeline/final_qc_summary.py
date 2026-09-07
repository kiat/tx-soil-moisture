"""Build final QC summaries after staged soil-gap filling.

This script does not modify station data. It audits an explicitly selected
pipeline stage, reports residual issues, and joins the recorded review decisions for
unavailable sensors, bound values, near-zero/flat sensors, and screened
very-long-gap segments.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from param_config import ALL_SOIL_PARAMS
from sensor_qc_decisions import validate_candidate_authorizations
from time_index_utils import require_unique_datetime_index
from soil_source_coverage import load_soil_coverage


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "final_qc_reports"
REVIEW_DECISIONS = BASE_DIR / "soil_qc_review_decisions.csv"
MANUAL_QC_MASKS = BASE_DIR / "manual_qc_masks.csv"
SENSOR_CANDIDATES = BASE_DIR / "sensor_qc_reports" / "sensor_qc_decisions.csv"

PARAM_SUMMARY_COLUMNS = [
    "Station", "Parameter", "Input File", "Start", "End", "Total Hours",
    "NaN Runs", "Max NaN Run Hours", "Short NaN Runs", "Medium NaN Runs",
    "Long NaN Runs", "VeryLong NaN Runs", "Flags", "Nonmissing Hours",
    "NaN Hours", "NaN Fraction", "Min", "Max", "Mean", "Std", "Range",
    "Exact Lower Bound Count", "Exact Upper Bound Count", "SWC Near-Zero Count",
    "SWC Near-Zero Fraction", "Longest Constant Run Hours",
    "Longest Constant Run Value",
    "Source Coverage Status", "Source Start", "Source End", "Inside Source Coverage Hours",
    "Outside Source Coverage NaN Hours", "Coverage Baseline SHA256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize final QC issues after gap filling.")
    parser.add_argument("--station", type=str, nargs="*", help="Station IDs/site codes to audit.")
    parser.add_argument("--param", type=str, nargs="*", help="Parameters to audit.")
    parser.add_argument(
        "--input-stage",
        choices=["verylong-repaired", "post-qc", "final"],
        default="final",
        help="Exact pipeline stage to audit; inputs never fall back to another stage.",
    )
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR, help="Directory for QC report CSVs.")
    parser.add_argument("--swc-near-zero", type=float, default=0.01)
    parser.add_argument("--swc-flat-range", type=float, default=0.01)
    parser.add_argument("--temp-flat-range", type=float, default=1.0)
    parser.add_argument("--dominant-fraction", type=float, default=0.5)
    parser.add_argument("--min-sensor-hours", type=int, default=720)
    return parser.parse_args()


def manual_qc_stations(path: Path = MANUAL_QC_MASKS) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Manual QC decision file not found: {path}")
    masks = pd.read_csv(path)
    required = {"Station", "Decision"}
    missing = required - set(masks.columns)
    if missing:
        raise ValueError(f"Manual QC decision file is missing columns: {sorted(missing)}")
    selected = masks[masks["Decision"].eq("mask_and_refill")].copy()
    return set(selected["Station"].astype(str))


def input_path_for(
    station: str,
    input_stage: str,
    stations_with_manual_qc: set[str] | None = None,
    directory: Path = OUT_DIR,
) -> Path:
    if input_stage == "verylong-repaired":
        suffix = "filled_verylonggaps_repaired.csv"
    elif input_stage == "final":
        suffix = "filled_final.csv"
    elif input_stage == "post-qc":
        if stations_with_manual_qc is None:
            raise ValueError("post-qc input requires the manual-QC station decision set")
        suffix = (
            "filled_manual_qc.csv"
            if station in stations_with_manual_qc
            else "filled_sensor_qc.csv"
        )
    else:
        raise ValueError(f"Unsupported final QC input stage: {input_stage}")

    path = Path(directory) / f"Station{station}_{suffix}"
    if not path.is_file():
        raise FileNotFoundError(
            f"final_qc_summary.py requires {input_stage} input for Station{station}: {path}"
        )
    return path


def discover_stations() -> List[str]:
    pat = re.compile(r"Station(.+)_filled_verylonggaps_repaired\.csv")
    return sorted(
        m.group(1)
        for path in OUT_DIR.glob("Station*_filled_verylonggaps_repaired.csv")
        if (m := pat.match(path.name))
    )


def read_station(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    require_unique_datetime_index(df, str(path))
    return df.sort_index()


def gap_category(hours: int) -> str:
    if hours < 24:
        return "short_<24h"
    if hours < 168:
        return "medium_24-167h"
    if hours < 720:
        return "long_168-719h"
    return "verylong_>=720h"


def nan_runs(df: pd.DataFrame, station: str, param: str, coverage=None) -> List[dict]:
    coverage = coverage or load_soil_coverage(station)
    return [{"Station": station, "Parameter": param, "Start": run[0], "End": run[-1],
             "Hours": len(run), "Category": gap_category(len(run))}
            for run in coverage.nan_runs(df, param)]


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


def summarize_verylong_review(
    stations: Iterable[str] | None = None,
    params: Iterable[str] | None = None,
) -> pd.DataFrame:
    path = BASE_DIR / "verylonggaps_review_segments.csv"
    if not path.exists():
        return pd.DataFrame()
    review = pd.read_csv(path)
    if review.empty:
        return pd.DataFrame()
    if stations is not None:
        review = review[review["Station"].astype(str).isin(set(stations))]
    if params is not None:
        review = review[review["Parameter"].isin(set(params))]
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


def load_review_decisions() -> pd.DataFrame:
    if not REVIEW_DECISIONS.exists():
        return pd.DataFrame()
    decisions = pd.read_csv(REVIEW_DECISIONS, parse_dates=["Start", "End"])
    required = {
        "Review Type", "Station", "Parameter", "Decision", "Action",
        "Decision Reason", "Status", "Review Date",
    }
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(
            f"{REVIEW_DECISIONS.name} is missing columns: {sorted(missing)}"
        )
    decisions["Station"] = decisions["Station"].astype(str)
    return decisions


def load_sensor_candidate_status(
    path: Path = SENSOR_CANDIDATES,
    required: bool = False,
) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Missing sensor candidate status file: {path}. Run the complete QC "
                "stage before post-QC or final QC reporting."
            )
        return pd.DataFrame()
    decisions = validate_candidate_authorizations(pd.read_csv(path), path.name)
    return decisions.loc[
        decisions["QC Decision"].eq("bad_sensor_candidate")
    ].copy()


def validate_diagnostic_scope(args: argparse.Namespace) -> None:
    """Keep filtered diagnostics out of the production report directory."""
    if not (args.station or args.param):
        return
    if args.report_dir.expanduser().resolve() == REPORT_DIR.resolve():
        raise ValueError(
            "Filtered final QC is diagnostic only. Supply an explicit non-production "
            "--report-dir so global production QC reports are not overwritten."
        )


def attach_sensor_decisions(
    suspicious: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "QC Decision", "Decision Action", "Decision Reason",
        "Review Status", "Review Date",
    ]
    if decisions.empty:
        for column in columns:
            suspicious[column] = pd.NA
        return suspicious

    sensor = decisions.loc[
        decisions["Review Type"].eq("final_sensor_flag"),
        [
            "Station", "Parameter", "Decision", "Action",
            "Decision Reason", "Status", "Review Date",
        ],
    ].rename(
        columns={
            "Decision": "QC Decision",
            "Action": "Decision Action",
            "Status": "Review Status",
        }
    )
    return suspicious.merge(
        sensor,
        on=["Station", "Parameter"],
        how="left",
        validate="one_to_one",
    )


def unresolved_review_count(
    suspicious: pd.DataFrame,
    decisions: pd.DataFrame,
    sensor_candidates: pd.DataFrame | None = None,
) -> int:
    review_status = suspicious.get(
        "Review Status",
        pd.Series(index=suspicious.index, dtype=object),
    ).fillna("unresolved")
    unresolved_sensor_rows = suspicious.loc[review_status.ne("closed")]
    if sensor_candidates is not None and not sensor_candidates.empty:
        candidate_keys = set(
            map(
                tuple,
                sensor_candidates[["Station", "Parameter"]].astype(str).to_numpy(),
            )
        )
        unresolved_sensor_rows = unresolved_sensor_rows.loc[
            ~unresolved_sensor_rows[["Station", "Parameter"]]
            .astype(str)
            .apply(tuple, axis=1)
            .isin(candidate_keys)
        ]
    unresolved_sensor = int(len(unresolved_sensor_rows))
    review_path = BASE_DIR / "verylonggaps_review_segments.csv"
    if not review_path.exists():
        unresolved_verylong = 0
    else:
        current = pd.read_csv(review_path, parse_dates=["Start", "End"])
        current = current.loc[current["Status"].eq("review")]
        if current.empty:
            unresolved_verylong = 0
        elif decisions.empty:
            unresolved_verylong = len(current)
        else:
            verylong = decisions.loc[
                decisions["Review Type"].eq("verylong_segment"),
                ["Station", "Parameter", "Start", "End", "Status"],
            ].rename(columns={"Status": "Review Status"})
            closure = current.merge(
                verylong,
                on=["Station", "Parameter", "Start", "End"],
                how="left",
                validate="one_to_one",
            )
            unresolved_verylong = int(
                closure["Review Status"].fillna("unresolved").ne("closed").sum()
            )

    unresolved_candidates = 0
    if sensor_candidates is not None and not sensor_candidates.empty:
        unresolved_candidates = int(
            sensor_candidates["Approval Status"].eq("pending").sum()
        )
    return unresolved_sensor + unresolved_verylong + unresolved_candidates


def write_outputs(
    overview_rows: List[dict],
    param_rows: List[dict],
    gap_rows: List[dict],
    missing_column_rows: List[dict],
    suspicious_rows: List[dict],
    report_dir: Path,
    stations: Iterable[str],
    params: Iterable[str],
    sensor_candidates: pd.DataFrame | None = None,
    coverage_rows: List[dict] | None = None,
) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    overview = pd.DataFrame(
        overview_rows,
        columns=["Station", "Input File", "Status", "Rows", "Remaining Soil NaN Hours", "Missing Soil Columns"],
    )
    param_summary = pd.DataFrame(param_rows, columns=PARAM_SUMMARY_COLUMNS)
    remaining_gaps = pd.DataFrame(
        gap_rows,
        columns=["Station", "Parameter", "Start", "End", "Hours", "Category"],
    )
    missing_columns = pd.DataFrame(
        missing_column_rows,
        columns=["Station", "Parameter", "Reason"],
    )
    suspicious = pd.DataFrame(suspicious_rows, columns=param_summary.columns)
    decisions = load_review_decisions()
    suspicious = attach_sensor_decisions(suspicious, decisions)
    verylong_review = summarize_verylong_review(stations, params)
    if verylong_review.empty:
        verylong_review = pd.DataFrame(
            columns=[
                "Station", "Parameter", "Status", "Review Reason", "Segments",
                "Filled_Hours", "Repaired_Points", "Max_Hourly_Change",
                "Max_Donor_Mean_Fraction",
            ]
        )

    overview.to_csv(report_dir / "final_qc_overview.csv", index=False)
    param_summary.to_csv(report_dir / "final_qc_station_parameter_summary.csv", index=False)
    remaining_gaps.to_csv(report_dir / "final_qc_remaining_nan_runs.csv", index=False)
    if coverage_rows is not None:
        pd.DataFrame(coverage_rows).to_csv(report_dir / "final_qc_soil_source_coverage.csv", index=False)
    missing_columns.to_csv(report_dir / "final_qc_missing_sensor_columns.csv", index=False)
    suspicious.to_csv(report_dir / "final_qc_suspicious_sensors.csv", index=False)
    verylong_review.to_csv(report_dir / "final_qc_verylong_review_summary.csv", index=False)
    if sensor_candidates is not None:
        sensor_candidates.to_csv(
            report_dir / "final_qc_sensor_candidate_status.csv",
            index=False,
        )
    if not decisions.empty:
        decisions.to_csv(report_dir / "final_qc_review_decisions.csv", index=False)
        (
            decisions.groupby(["Review Type", "Status", "Decision"], dropna=False)
            .agg(
                Items=("Decision", "size"),
                Values_Changed=("Values Changed", "sum"),
            )
            .reset_index()
            .to_csv(report_dir / "final_qc_review_closure_summary.csv", index=False)
        )
    return unresolved_review_count(suspicious, decisions, sensor_candidates)


def main() -> None:
    args = parse_args()
    validate_diagnostic_scope(args)
    stations = args.station if args.station else discover_stations()
    params = args.param if args.param else ALL_SOIL_PARAMS
    if not stations:
        raise FileNotFoundError(
            "No validated very-long-gap station cohort was found; specify --station "
            "or run validate_verylonggaps.py --write-repaired first."
        )
    stations_with_manual_qc = (
        manual_qc_stations() if args.input_stage == "post-qc" else set()
    )
    sensor_candidates = (
        load_sensor_candidate_status(required=True)
        if args.input_stage in {"post-qc", "final"}
        else None
    )
    if sensor_candidates is not None and not sensor_candidates.empty:
        sensor_candidates = sensor_candidates.loc[
            sensor_candidates["Station"].astype(str).isin(stations)
            & sensor_candidates["Parameter"].astype(str).isin(params)
        ].copy()

    overview_rows: List[dict] = []
    param_rows: List[dict] = []
    gap_rows: List[dict] = []
    missing_column_rows: List[dict] = []
    suspicious_rows: List[dict] = []
    coverage_rows: List[dict] = []

    for station in stations:
        path = input_path_for(station, args.input_stage, stations_with_manual_qc)
        df = read_station(path)
        coverage = load_soil_coverage(station)
        coverage.assert_frame(df)
        source_records = {row["Parameter"]: row for row in coverage.records(df)}
        coverage_rows.extend(source_records[param] for param in params)
        station_nan = 0

        for param in params:
            if param not in df.columns:
                missing_column_rows.append({"Station": station, "Parameter": param, "Reason": "column_missing"})
                continue

            runs = nan_runs(df, station, param, coverage)
            gap_rows.extend(runs)
            nan_hours = sum(int(r["Hours"]) for r in runs)
            station_nan += nan_hours
            run_counts = pd.Series([r["Category"] for r in runs]).value_counts().to_dict() if runs else {}

            flags, metrics = sensor_flags(param, df.loc[coverage.contains(df.index, param), param], args)
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
            summary.update(source_records[param])
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

    unresolved_reviews = write_outputs(
        overview_rows,
        param_rows,
        gap_rows,
        missing_column_rows,
        suspicious_rows,
        args.report_dir,
        stations,
        params,
        sensor_candidates,
        coverage_rows,
    )

    remaining = pd.DataFrame(gap_rows)
    suspicious = pd.DataFrame(suspicious_rows)
    missing_columns = pd.DataFrame(missing_column_rows)

    print("Final QC summary complete.")
    print(f"Stations audited: {len(stations)}")
    print(f"Remaining NaN runs: {len(remaining)}")
    print(f"Remaining NaN hours: {int(remaining['Hours'].sum()) if not remaining.empty else 0}")
    print(f"Missing sensor columns: {len(missing_columns)}")
    print(f"Suspicious station/parameter rows: {len(suspicious)}")
    pending_candidates = (
        int(sensor_candidates["Approval Status"].eq("pending").sum())
        if sensor_candidates is not None
        else 0
    )
    print(f"Unresolved whole-sensor candidates: {pending_candidates}")
    print(f"Unresolved recorded review items: {unresolved_reviews}")
    print(f"Outputs written under: {args.report_dir}")


if __name__ == "__main__":
    main()

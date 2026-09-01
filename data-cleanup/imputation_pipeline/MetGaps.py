#!/usr/bin/env python3
"""Audit MET availability and fill internal MET gaps.

This stage is intentionally separate from the soil workflow. It reads the
Stage 0 cleaned files and writes MET-only outputs, so existing soil outputs are
never overwritten. Valid dedicated-MET precipitation is preferred with valid
same-station soil precipitation as fallback.

The default mode conservatively interpolates only bracketed gaps shorter than
24 hours. ``--full`` uses the expanded five-gap benchmark method map for every
internal Tair, RH, Srad, wind-speed, and wind-direction gap. The optional
``--repair-review`` pass masks high-confidence sensor faults and
repairs parameter-specific boundary/physical QC failures. Ppt source
reconciliation follows the project-approved MET-first rule; model filling for
hours missing from all direct sources remains a separate stage.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from datacleaning import load_met_data, load_soil_data
from param_config import ALL_MET_PARAMS, short_interp_for


BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "cleaned_data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "met_output"
DEFAULT_REPORT_DIR = BASE_DIR / "met_qc_reports"
DEFAULT_RAW_DATA_DIR = BASE_DIR.parents[1] / "datasets" / "TxSON_data_2026-02-24"

PHYSICAL_BOUNDS = {
    "Ppt": (0.0, None),
    "Tair": (-30.0, 60.0),
    "RH": (0.0, 100.0),
    "Srad": (0.0, None),
    "Wind speed": (0.0, 25.0),
    "Wind direction": (0.0, 360.0),
}

NON_PPT_MET_PARAMS = [
    "Tair", "RH", "Srad", "Wind speed", "Wind direction",
]

# Winners retained after benchmark validation. Short-gap Wind direction was
# also confirmed with a second random seed before changing the production map.
MET_GAP_METHODS = {
    "Tair": {
        "short": "donor_regression",
        "medium": "donor_regression",
        "long": "donor_regression",
        "verylong": "donor_regression",
    },
    "RH": {
        "short": "linear",
        "medium": "xgboost",
        "long": "donor_regression",
        "verylong": "donor_regression",
    },
    "Srad": {
        "short": "donor_regression",
        "medium": "random_forest",
        "long": "xgboost",
        "verylong": "donor_regression",
    },
    "Wind speed": {
        "short": "xgboost",
        "medium": "xgboost",
        "long": "xgboost",
        "verylong": "donor_regression",
    },
    "Wind direction": {
        "short": "random_forest",
        "medium": "random_forest",
        "long": "random_forest",
        "verylong": "xgboost",
    },
}

MODEL_RANDOM_SEED = 42
MAX_TRAIN_ROWS = 50_000
PPT_MODEL_NAME = "two_part_random_forest"
PPT_FALLBACK_RAIN_PROBABILITY_THRESHOLD = 0.5
PPT_UNCERTAIN_THRESHOLD_MARGIN = 0.05

# Simple screening thresholds, not automatic rejection rules. Predictions stay
# in the output so the hourly series remains complete, while the segment report
# clearly separates accepted fills from values needing visual/domain review.
HOURLY_JUMP_REVIEW_LIMITS = {
    "Tair": 8.0,
    "RH": 30.0,
    "Srad": 500.0,
    "Wind speed": 5.0,
    "Wind direction": 120.0,
}
# Boundary and within-gap hourly jumps use the same parameter-specific screen.
BOUNDARY_REVIEW_LIMITS = HOURLY_JUMP_REVIEW_LIMITS

# High-confidence sensor-QC rules. Network rules require at least three
# simultaneously reporting stations. A station-level RH rule also catches a
# persistent, repeated low-scale state without discarding isolated dry hours.
# Uncertain observations are retained for review rather than removed.
SENSOR_QC_MIN_NETWORK_STATIONS = 3
TAIR_NETWORK_RESIDUAL_LIMIT = 15.0
TAIR_DONOR_AGREEMENT_LIMIT = 8.0
TAIR_BOUNDARY_RESIDUAL_LIMIT = 10.0
TAIR_BOUNDARY_DONOR_AGREEMENT_LIMIT = 12.0
TAIR_BOUNDARY_EXPANSION_HOURS = 24
RH_LOW_VALUE_LIMIT = 10.0
RH_LOW_SCALE_LIMIT = 5.0
RH_LOW_SCALE_MIN_HOURS = 100
RH_LOW_SCALE_MIN_SHARE = 0.75
RH_LOW_SCALE_MIN_RUN = 12
RH_HIGH_VALUE_LIMIT = 99.5
RH_NETWORK_MEDIAN_MIN = 30.0
RH_NETWORK_DONOR_FLOOR = 15.0
RH_NETWORK_RESIDUAL_MIN = 25.0
RH_DONOR_AGREEMENT_LIMIT = 30.0
RH_EXTREME_NETWORK_RESIDUAL_LIMIT = 50.0
RH_EXTREME_NETWORK_MIN_DONORS = 3
RH_BOUNDARY_DONOR_AGREEMENT_LIMIT = 12.0
RH_BOUNDARY_RESIDUAL_MIN = 30.0

PPT_COMPARISON_COLUMNS = [
    "Station",
    "Status",
    "Soil Source Hours",
    "MET Source Hours",
    "Overlapping Valid Hours",
    "Matching Hours",
    "Different Hours",
    "Different Percent of Overlap",
    "Wet Dry Disagreement Hours",
    "Mean Absolute Difference",
    "Maximum Absolute Difference",
    "Soil Total on Overlap",
    "MET Total on Overlap",
    "MET Minus Soil Total",
    "Reconciled From MET Hours",
    "Reconciled From Soil Fallback Hours",
    "Additional Hours Recovered",
    "Remaining Missing Hours",
    "Stage 0 Mismatch Hours",
]

MODEL_DETAIL_COLUMNS = [
    "Station",
    "Parameter",
    "Gap Class",
    "Start",
    "End",
    "Hours",
    "Method",
    "Fallback",
    "Action",
    "Filled Hours",
    "Remaining Hours",
    "Raw Physical Violations",
    "Left Boundary Jump",
    "Right Boundary Jump",
    "Max Internal Hourly Jump",
    "Original QC Reason",
    "Repair Method",
    "QC Status",
    "QC Reason",
    "Error",
]

SENSOR_QC_REPORT_COLUMNS = [
    "Station",
    "Parameter",
    "Start",
    "End",
    "Masked Hours",
    "Rule",
    "Evidence Seed Hours",
    "Target Min",
    "Target Max",
    "Network Median",
    "Network Span",
    "Median Absolute Residual",
]

PPT_MODEL_DETAIL_COLUMNS = [
    "Station",
    "Gap Class",
    "Start",
    "End",
    "Hours",
    "Internal Gap",
    "Method",
    "Rain Probability Threshold",
    "Action",
    "Predicted Wet Hours",
    "Predicted Total Ppt",
    "Median Donor Count",
    "Three Plus Donor Wet Hours",
    "Max Donor Wet Count",
    "Zero Donor Hours",
    "Uncertain Occurrence Hours",
    "Max Rain Probability",
    "Threshold Margin",
    "Remaining Hours",
    "QC Status",
    "QC Reason",
    "Review Priority",
    "Error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit MET data and reconcile precipitation sources. The default "
            "mode fills bracketed gaps shorter than 24 hours; --full fills all "
            "internal non-precipitation MET gaps with the benchmark method map; "
            "--ppt-full fills precipitation missing from all direct sources."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--station", type=str, nargs="*", help="Station/site codes; omit to discover cleaned files.")
    parser.add_argument("--param", type=str, nargs="*", help="MET parameters; omit for all configured MET parameters.")
    parser.add_argument("--max-gap", type=int, default=24, help="Exclusive short-gap upper bound in hours.")
    parser.add_argument("--write", action="store_true", help="Write MET-only station outputs.")
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Fill all internal non-precipitation MET gaps with the benchmark "
            "method map. Without this flag, retain the conservative short-gap stage."
        ),
    )
    parser.add_argument(
        "--repair-review",
        action="store_true",
        help=(
            "With --full, mask high-confidence Tair/RH sensor faults and "
            "repair parameter-specific segment QC failures."
        ),
    )
    parser.add_argument(
        "--ppt-full",
        action="store_true",
        help=(
            "Reconcile all direct Ppt sources, then fill hours missing from all "
            "approved sources with the benchmark-selected two-part Random Forest."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--soil-base-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Raw soil-file directory used for precipitation reconciliation.",
    )
    parser.add_argument(
        "--met-base-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Raw dedicated-MET directory used for precipitation reconciliation.",
    )
    return parser.parse_args()


def discover_stations(directory: Path = CLEAN_DIR) -> List[str]:
    pattern = re.compile(r"Station(.+)_cleaned_data\.csv")
    return sorted(
        match.group(1)
        for path in directory.glob("Station*_cleaned_data.csv")
        if (match := pattern.fullmatch(path.name))
    )


def discover_met_source_stations(directory: Path = CLEAN_DIR) -> List[str]:
    stations = []
    for station in discover_stations(directory):
        path = directory / f"Station{station}_cleaned_data.csv"
        columns = pd.read_csv(path, nrows=0).columns
        if any(parameter in columns for parameter in NON_PPT_MET_PARAMS):
            stations.append(station)
    return stations


def read_cleaned_station(station: str, directory: Path = CLEAN_DIR) -> pd.DataFrame:
    path = directory / f"Station{station}_cleaned_data.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.empty:
        return df
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", name="Date"))


def preserve_unselected_output(
    result: pd.DataFrame,
    output_path: Path,
    selected: Iterable[str],
) -> pd.DataFrame:
    """Keep previously filled parameters during a targeted parameter rerun."""
    if not output_path.exists():
        return result
    existing = pd.read_csv(output_path, index_col=0, parse_dates=True, low_memory=False)
    existing.index = pd.DatetimeIndex(existing.index)
    existing.index.name = "Date"
    existing = existing[~existing.index.duplicated(keep="first")].sort_index()
    merged = existing.reindex(result.index).copy()
    for parameter in selected:
        if parameter in result:
            merged[parameter] = result[parameter]
    for column in result.columns:
        if column not in merged:
            merged[column] = result[column]
    merged.index.name = "Date"
    return merged


def valid_precipitation(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    values = pd.to_numeric(series.reindex(index), errors="coerce")
    return values.where(values >= 0.0)


def reconcile_ppt_series(
    station: str,
    index: pd.DatetimeIndex,
    soil_ppt: pd.Series,
    met_ppt: pd.Series,
    stage0_ppt: pd.Series,
) -> Tuple[pd.Series, dict]:
    """Use valid dedicated-MET rain first and same-station soil rain as fallback."""
    soil = valid_precipitation(soil_ppt, index)
    met = valid_precipitation(met_ppt, index)
    current = valid_precipitation(stage0_ppt, index)
    reconciled = met.combine_first(soil)

    overlap = soil.notna() & met.notna()
    matching = pd.Series(False, index=index)
    matching.loc[overlap] = np.isclose(
        soil.loc[overlap].to_numpy(),
        met.loc[overlap].to_numpy(),
        rtol=0.0,
        atol=1e-6,
    )
    different = overlap & ~matching
    wet_dry_disagreement = overlap & ((soil > 0.0) != (met > 0.0))
    absolute_difference = (met.loc[overlap] - soil.loc[overlap]).abs()

    current_matches = (current.isna() & reconciled.isna()) | (
        current.notna()
        & reconciled.notna()
        & np.isclose(current, reconciled, rtol=0.0, atol=1e-6)
    )
    overlap_hours = int(overlap.sum())
    different_hours = int(different.sum())
    row = {
        "Station": station,
        "Status": "compared",
        "Soil Source Hours": int(soil.notna().sum()),
        "MET Source Hours": int(met.notna().sum()),
        "Overlapping Valid Hours": overlap_hours,
        "Matching Hours": int((overlap & matching).sum()),
        "Different Hours": different_hours,
        "Different Percent of Overlap": (
            100.0 * different_hours / overlap_hours if overlap_hours else np.nan
        ),
        "Wet Dry Disagreement Hours": int(wet_dry_disagreement.sum()),
        "Mean Absolute Difference": (
            float(absolute_difference.mean()) if overlap_hours else np.nan
        ),
        "Maximum Absolute Difference": (
            float(absolute_difference.max()) if overlap_hours else np.nan
        ),
        "Soil Total on Overlap": float(soil.loc[overlap].sum()),
        "MET Total on Overlap": float(met.loc[overlap].sum()),
        "MET Minus Soil Total": float((met.loc[overlap] - soil.loc[overlap]).sum()),
        "Reconciled From MET Hours": int(met.notna().sum()),
        "Reconciled From Soil Fallback Hours": int((met.isna() & soil.notna()).sum()),
        "Additional Hours Recovered": int((current.isna() & reconciled.notna()).sum()),
        "Remaining Missing Hours": int(reconciled.isna().sum()),
        "Stage 0 Mismatch Hours": int((~current_matches).sum()),
    }
    reconciled.index.name = "Date"
    return reconciled, row


def reconcile_station_ppt(
    station: str,
    cleaned: pd.DataFrame,
    soil_base_dir: Path,
    met_base_dir: Path,
) -> Tuple[pd.Series, dict]:
    current = pd.to_numeric(cleaned["Ppt"], errors="coerce")
    station_met = met_base_dir / f"{station}_met.dat"
    legacy_met = met_base_dir / f"MET_{station}.dat"
    if not station_met.exists() and not legacy_met.exists():
        return current, {
            "Station": station,
            "Status": "no_dedicated_met",
            "Soil Source Hours": int(current.notna().sum()),
            "Remaining Missing Hours": int(current.isna().sum()),
        }

    soil = load_soil_data(station, soil_base_dir)
    met = load_met_data(station, met_base_dir)
    if "Ppt" not in soil.columns or "Ppt" not in met.columns:
        return current, {
            "Station": station,
            "Status": "ppt_column_missing",
            "Remaining Missing Hours": int(current.isna().sum()),
        }
    return reconcile_ppt_series(
        station,
        cleaned.index,
        soil["Ppt"],
        met["Ppt"],
        current,
    )


def contiguous_nan_runs(series: pd.Series) -> List[pd.DatetimeIndex]:
    missing = series.isna().to_numpy()
    runs: List[pd.DatetimeIndex] = []
    i = 0
    while i < len(missing):
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < len(missing) and missing[j]:
            j += 1
        runs.append(pd.DatetimeIndex(series.index[i:j]))
        i = j
    return runs


def contiguous_true_runs(mask: pd.Series) -> List[pd.DatetimeIndex]:
    values = mask.fillna(False).to_numpy(dtype=bool)
    runs: List[pd.DatetimeIndex] = []
    i = 0
    while i < len(values):
        if not values[i]:
            i += 1
            continue
        j = i
        while j < len(values) and values[j]:
            j += 1
        runs.append(pd.DatetimeIndex(mask.index[i:j]))
        i = j
    return runs


def suspicious_low_scale_rh_mask(target: pd.Series) -> pd.Series:
    """Find a persistent low-scale RH state while preserving isolated lows."""
    low_scale = target.notna() & target.lt(RH_LOW_SCALE_LIMIT)
    sub_ten = target.notna() & target.lt(RH_LOW_VALUE_LIMIT)
    low_hours = int(low_scale.sum())
    sub_ten_hours = int(sub_ten.sum())
    longest_run = max(
        (len(run) for run in contiguous_true_runs(low_scale)),
        default=0,
    )
    suspicious = (
        low_hours >= RH_LOW_SCALE_MIN_HOURS
        and sub_ten_hours > 0
        and low_hours / sub_ten_hours >= RH_LOW_SCALE_MIN_SHARE
        and longest_run >= RH_LOW_SCALE_MIN_RUN
    )
    if suspicious:
        return low_scale
    return pd.Series(False, index=target.index)


def expand_rh_mask_at_bad_gap_boundaries(
    target: pd.Series,
    initial_mask: pd.Series,
    network_count: pd.Series,
    network_span: pd.Series,
    residual: pd.Series,
) -> pd.Series:
    """Mask only donor-confirmed bad boundaries around short RH gaps."""
    mask = initial_mask.copy()
    for _ in range(24):
        working = target.mask(mask)
        additions = pd.Series(False, index=target.index)
        for run in contiguous_nan_runs(working):
            if len(run) >= 24:
                continue
            left = run[0] - pd.Timedelta(hours=1)
            right = run[-1] + pd.Timedelta(hours=1)
            if (
                left not in working.index
                or right not in working.index
                or pd.isna(working.loc[left])
                or pd.isna(working.loc[right])
            ):
                continue
            linear_step = abs(
                float(working.loc[right]) - float(working.loc[left])
            ) / (len(run) + 1)
            if linear_step <= BOUNDARY_REVIEW_LIMITS["RH"]:
                continue
            for boundary in (left, right):
                if (
                    network_count.loc[boundary]
                    >= SENSOR_QC_MIN_NETWORK_STATIONS - 1
                    and network_span.loc[boundary]
                    <= RH_BOUNDARY_DONOR_AGREEMENT_LIMIT
                    and residual.loc[boundary] > RH_BOUNDARY_RESIDUAL_MIN
                ):
                    additions.loc[boundary] = True
        new_values = additions & ~mask
        if not new_values.any():
            break
        mask |= new_values
    return mask


def expand_tair_mask_around_confirmed_faults(
    target: pd.Series,
    initial_mask: pd.Series,
    network_count: pd.Series,
    network_span: pd.Series,
    residual: pd.Series,
) -> pd.Series:
    """Include donor-confirmed edge values next to a Tair fault run."""
    mask = initial_mask.copy()
    for _ in range(TAIR_BOUNDARY_EXPANSION_HOURS):
        additions = pd.Series(False, index=target.index)
        for run in contiguous_true_runs(mask):
            for boundary in (
                run[0] - pd.Timedelta(hours=1),
                run[-1] + pd.Timedelta(hours=1),
            ):
                if boundary not in target.index or pd.isna(target.loc[boundary]):
                    continue
                if (
                    network_count.loc[boundary]
                    >= SENSOR_QC_MIN_NETWORK_STATIONS - 1
                    and network_span.loc[boundary]
                    <= TAIR_BOUNDARY_DONOR_AGREEMENT_LIMIT
                    and residual.loc[boundary] > TAIR_BOUNDARY_RESIDUAL_LIMIT
                ):
                    additions.loc[boundary] = True
        new_values = additions & ~mask
        if not new_values.any():
            break
        mask |= new_values
    return mask


def apply_network_sensor_qc(
    source_frames: Dict[str, pd.DataFrame],
    parameters: Iterable[str],
) -> Tuple[Dict[str, pd.DataFrame], List[dict], Dict[Tuple[str, str], int]]:
    """Mask only high-confidence Tair/RH faults supported by network data."""
    repaired = {station: frame.copy() for station, frame in source_frames.items()}
    detail_rows: List[dict] = []
    mask_counts: Dict[Tuple[str, str], int] = {}

    for parameter in [p for p in parameters if p in {"Tair", "RH"}]:
        columns = {
            station: pd.to_numeric(frame[parameter], errors="coerce")
            for station, frame in source_frames.items()
            if parameter in frame.columns
        }
        if len(columns) < SENSOR_QC_MIN_NETWORK_STATIONS:
            continue
        network = pd.concat(columns, axis=1).sort_index()
        full_index = pd.date_range(network.index.min(), network.index.max(), freq="h")
        network = network.reindex(full_index)
        donor_network = network
        if parameter == "RH":
            donor_network = network.copy()
            for donor_station in donor_network.columns:
                donor_low_scale = suspicious_low_scale_rh_mask(
                    donor_network[donor_station]
                )
                donor_network.loc[donor_low_scale, donor_station] = np.nan
        for station, target in network.items():
            donors = donor_network.drop(columns=station)
            network_median = donors.median(axis=1, skipna=True)
            network_count = donors.notna().sum(axis=1)
            network_min = donors.min(axis=1, skipna=True)
            network_span = donors.max(axis=1) - donors.min(axis=1)
            residual = (target - network_median).abs()
            enough_network = network_count >= max(
                SENSOR_QC_MIN_NETWORK_STATIONS - 1,
                1,
            )
            if parameter == "Tair":
                seed = (
                    target.notna()
                    & enough_network
                    & network_span.le(TAIR_DONOR_AGREEMENT_LIMIT)
                    & residual.gt(TAIR_NETWORK_RESIDUAL_LIMIT)
                )
                mask = expand_tair_mask_around_confirmed_faults(
                    target,
                    seed,
                    network_count,
                    network_span,
                    residual,
                )
                trigger_masks = {
                    "network_tair_residual_gt_15C": seed,
                    "donor_confirmed_tair_fault_boundary": mask & ~seed,
                }
            else:
                low_scale_rail = suspicious_low_scale_rh_mask(target)
                extreme_network_residual = (
                    target.notna()
                    & network_count.ge(RH_EXTREME_NETWORK_MIN_DONORS)
                    & network_span.le(RH_BOUNDARY_DONOR_AGREEMENT_LIMIT)
                    & residual.gt(RH_EXTREME_NETWORK_RESIDUAL_LIMIT)
                )
                low_rail = (
                    target.notna()
                    & enough_network
                    & target.lt(RH_LOW_VALUE_LIMIT)
                    & (
                        (
                            network_span.le(RH_DONOR_AGREEMENT_LIMIT)
                            & network_median.gt(RH_NETWORK_MEDIAN_MIN)
                            & residual.gt(RH_NETWORK_RESIDUAL_MIN)
                        )
                        | network_min.gt(RH_NETWORK_DONOR_FLOOR)
                    )
                )
                high_rail = (
                    target.notna()
                    & enough_network
                    & network_span.le(RH_DONOR_AGREEMENT_LIMIT)
                    & target.ge(RH_HIGH_VALUE_LIMIT)
                    & residual.gt(RH_NETWORK_RESIDUAL_MIN)
                )
                low_value = target.lt(RH_LOW_VALUE_LIMIT)
                rapid_low_transition = pd.Series(False, index=target.index)
                for lag in (1, 2, 3):
                    # When donor coverage is temporarily insufficient, a
                    # same-sensor jump from the low rail to >60% within three
                    # hours is still strong evidence against the low endpoint.
                    rapid_low_transition |= low_value & (
                        target.shift(lag).gt(60.0)
                        | target.shift(-lag).gt(60.0)
                    )
                # Keep normal-looking observations between intermittent rail
                # failures. Broad episode expansion removed too many values
                # that still agreed with the network.
                seed = (
                    low_scale_rail
                    | extreme_network_residual
                    | low_rail
                    | high_rail
                    | rapid_low_transition
                )
                mask = expand_rh_mask_at_bad_gap_boundaries(
                    target,
                    seed,
                    network_count,
                    network_span,
                    residual,
                )
                trigger_masks = {
                    "persistent_low_scale_rh": low_scale_rail,
                    "extreme_network_rh_residual": extreme_network_residual,
                    "donor_confirmed_low_rh": low_rail,
                    "donor_confirmed_high_rh": high_rail,
                    "rapid_low_to_normal_transition": rapid_low_transition,
                    "donor_confirmed_bad_gap_boundary": mask & ~seed,
                }

            station_index = repaired[station].index
            station_mask = mask.reindex(station_index, fill_value=False)
            mask_counts[(station, parameter)] = int(station_mask.sum())
            if not station_mask.any():
                continue

            station_target = target.reindex(station_index)
            station_network = network_median.reindex(station_index)
            station_residual = residual.reindex(station_index)
            station_seed = seed.reindex(station_index, fill_value=False)
            station_trigger_masks = {
                name: trigger_mask.reindex(
                    station_index,
                    fill_value=False,
                )
                for name, trigger_mask in trigger_masks.items()
            }
            for run in contiguous_true_runs(station_mask):
                rules = [
                    name
                    for name, trigger_mask in station_trigger_masks.items()
                    if trigger_mask.loc[run].any()
                ]
                detail_rows.append(
                    {
                        "Station": station,
                        "Parameter": parameter,
                        "Start": run[0],
                        "End": run[-1],
                        "Masked Hours": len(run),
                        "Rule": ";".join(rules),
                        "Evidence Seed Hours": int(station_seed.loc[run].sum()),
                        "Target Min": float(station_target.loc[run].min()),
                        "Target Max": float(station_target.loc[run].max()),
                        "Network Median": float(station_network.loc[run].median()),
                        "Network Span": float(
                            network_span.reindex(station_index).loc[run].median()
                        ),
                        "Median Absolute Residual": float(
                            station_residual.loc[run].median()
                        ),
                    }
                )
            repaired[station].loc[station_mask, parameter] = np.nan

    return repaired, detail_rows, mask_counts


def gap_category(hours: int) -> str:
    if hours < 24:
        return "short_<24h"
    if hours < 168:
        return "medium_24-167h"
    if hours < 720:
        return "long_168-719h"
    return "verylong_>=720h"


def gap_class(hours: int) -> str:
    if hours < 24:
        return "short"
    if hours < 168:
        return "medium"
    if hours < 720:
        return "long"
    return "verylong"


def is_internal(series: pd.Series, run: pd.DatetimeIndex) -> bool:
    left = run[0] - pd.Timedelta(hours=1)
    right = run[-1] + pd.Timedelta(hours=1)
    return (
        left in series.index
        and right in series.index
        and pd.notna(series.loc[left])
        and pd.notna(series.loc[right])
    )


def interpolate_internal_run(series: pd.Series, run: pd.DatetimeIndex, method: str) -> pd.Series:
    left = run[0] - pd.Timedelta(hours=1)
    right = run[-1] + pd.Timedelta(hours=1)
    window = series.loc[left:right].astype(float)

    if method == "wind_angle":
        radians = np.deg2rad(window)
        sin_values = pd.Series(np.sin(radians), index=window.index).interpolate(method="time")
        cos_values = pd.Series(np.cos(radians), index=window.index).interpolate(method="time")
        values = (np.degrees(np.arctan2(sin_values, cos_values)) + 360.0) % 360.0
        return pd.Series(values, index=window.index).reindex(run)

    # MET short gaps use bracketed time interpolation. The configured "zero"
    # precipitation hint is intentionally not used in this dedicated stage.
    return window.interpolate(method="time").reindex(run)


def apply_bounds(values: pd.Series, parameter: str) -> pd.Series:
    lower, upper = PHYSICAL_BOUNDS[parameter]
    return values.clip(lower=lower, upper=upper)


def angular_jump(left: float, right: float) -> float:
    return float(abs((right - left + 180.0) % 360.0 - 180.0))


def boundary_jump(parameter: str, left: float, right: float) -> float:
    if parameter == "Wind direction":
        return angular_jump(left, right)
    return float(abs(right - left))


def build_model_features(
    station: str,
    parameter: str,
    target: pd.Series,
    source_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    index = target.index
    features = pd.DataFrame(index=index)
    hour = index.hour.to_numpy()
    day = index.dayofyear.to_numpy()
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["doy_sin"] = np.sin(2 * np.pi * day / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * day / 365.25)
    features["trend"] = np.arange(len(index), dtype=float) / max(len(index), 1)

    for lag in [1, 2, 3, 24, 48, 168]:
        features[f"target_lag_{lag}"] = target.shift(lag)

    local = source_frames[station]
    for column in ALL_MET_PARAMS:
        if column != parameter and column in local.columns:
            features[f"local__{column}"] = pd.to_numeric(
                local[column], errors="coerce"
            ).reindex(index)

    for donor, donor_frame in source_frames.items():
        if donor != station and parameter in donor_frame.columns:
            features[f"donor__{donor}"] = pd.to_numeric(
                donor_frame[parameter], errors="coerce"
            ).reindex(index)
    return features.replace([np.inf, -np.inf], np.nan)


def build_ppt_features(
    station: str,
    target: pd.Series,
    source_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build coverage-robust rainfall features for long sensor outages."""
    index = target.index
    hour = index.hour.to_numpy()
    day = index.dayofyear.to_numpy()
    features = pd.DataFrame(index=index)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["doy_sin"] = np.sin(2 * np.pi * day / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * day / 365.25)

    donors = pd.concat(
        {
            donor: pd.to_numeric(frame["Ppt"], errors="coerce").reindex(index)
            for donor, frame in source_frames.items()
            if donor != station and "Ppt" in frame
        },
        axis=1,
    )
    positive = donors.where(donors.gt(0.0))
    features["donor_available_count"] = donors.notna().sum(axis=1)
    features["donor_wet_count"] = donors.gt(0.0).sum(axis=1)
    features["donor_total"] = donors.clip(lower=0.0).sum(axis=1, min_count=1)
    features["donor_max"] = donors.max(axis=1)
    features["donor_positive_mean"] = positive.mean(axis=1)
    features["donor_positive_median"] = positive.median(axis=1)
    return features.replace([np.inf, -np.inf], np.nan)


def sample_training_rows(
    features: pd.DataFrame,
    target: pd.Series,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series]:
    positions = np.flatnonzero(target.notna().to_numpy())
    if len(positions) > MAX_TRAIN_ROWS:
        positions = np.sort(
            rng.choice(positions, size=MAX_TRAIN_ROWS, replace=False)
        )
    return features.iloc[positions], target.iloc[positions]


def make_tree_estimator(method: str, seed: int):
    if method == "random_forest":
        return RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    if method == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is unavailable")
        return XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Unsupported tree method: {method}")


def tree_prediction(
    method: str,
    features: pd.DataFrame,
    target: pd.Series,
    predict_index: pd.DatetimeIndex,
    parameter: str,
    rng: np.random.Generator,
) -> pd.Series:
    train_x, train_y = sample_training_rows(features, target, rng)
    if len(train_y) < 200:
        raise ValueError("insufficient training observations")
    test_x = features.reindex(predict_index)

    def pipeline(seed: int):
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            make_tree_estimator(method, seed),
        )

    if parameter == "Wind direction":
        radians = np.deg2rad(train_y.to_numpy(dtype=float))
        sin_model = pipeline(MODEL_RANDOM_SEED)
        cos_model = pipeline(MODEL_RANDOM_SEED + 1)
        sin_model.fit(train_x, np.sin(radians))
        cos_model.fit(train_x, np.cos(radians))
        values = (
            np.degrees(
                np.arctan2(
                    sin_model.predict(test_x),
                    cos_model.predict(test_x),
                )
            )
            + 360.0
        ) % 360.0
        return pd.Series(values, index=predict_index)

    model = pipeline(MODEL_RANDOM_SEED)
    model.fit(train_x, train_y)
    return pd.Series(model.predict(test_x), index=predict_index)


def select_occurrence_threshold(
    actual: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Select a station threshold by out-of-bag critical success index."""
    actual = np.asarray(actual, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(probability)
    if valid.sum() < 100 or not actual[valid].any():
        return PPT_FALLBACK_RAIN_PROBABILITY_THRESHOLD

    best_threshold = PPT_FALLBACK_RAIN_PROBABILITY_THRESHOLD
    best_score = (-np.inf, -np.inf, -np.inf)
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = probability[valid] >= threshold
        observed = actual[valid]
        true_positive = int((predicted & observed).sum())
        false_positive = int((predicted & ~observed).sum())
        false_negative = int((~predicted & observed).sum())
        denominator = true_positive + false_positive + false_negative
        csi = true_positive / denominator if denominator else 0.0
        recall_denominator = true_positive + false_negative
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        score = (csi, recall, -abs(threshold - 0.5))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def two_part_ppt_prediction(
    features: pd.DataFrame,
    target: pd.Series,
    predict_index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Predict rain occurrence first and positive rain amount second."""
    train_x, train_y = sample_training_rows(features, target, rng)
    if len(train_y) < 500:
        raise ValueError("insufficient Ppt training observations")

    occurrence = train_y.gt(0.0).astype(int)
    if occurrence.nunique() < 2:
        raise ValueError("Ppt training data has only one occurrence class")
    if int(occurrence.sum()) < 50:
        raise ValueError("insufficient positive-rain training observations")

    classifier = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            oob_score=True,
            random_state=MODEL_RANDOM_SEED,
            n_jobs=-1,
        ),
    )
    amount_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=MODEL_RANDOM_SEED,
            n_jobs=-1,
        ),
    )
    classifier.fit(train_x, occurrence)
    fitted_classifier = classifier.named_steps["randomforestclassifier"]
    threshold = select_occurrence_threshold(
        occurrence.to_numpy(dtype=bool),
        fitted_classifier.oob_decision_function_[:, 1],
    )
    wet = train_y.gt(0.0)
    amount_model.fit(train_x.loc[wet], np.log1p(train_y.loc[wet]))

    test_x = features.reindex(predict_index)
    probability = classifier.predict_proba(test_x)[:, 1]
    positive_amount = np.expm1(amount_model.predict(test_x)).clip(min=0.0)
    prediction = np.where(
        probability >= threshold,
        positive_amount,
        0.0,
    )
    return pd.DataFrame(
        {
            "Prediction": prediction,
            "Rain Probability": probability,
            "Rain Threshold": threshold,
        },
        index=predict_index,
    )


def donor_regression_prediction(
    features: pd.DataFrame,
    target: pd.Series,
    predict_index: pd.DatetimeIndex,
) -> pd.Series:
    donors = features.filter(regex=r"^donor__")
    if donors.empty:
        raise ValueError("no donor features")
    donor_mean = donors.mean(axis=1)
    valid = target.notna() & donor_mean.notna()
    if int(valid.sum()) < 200:
        raise ValueError("insufficient donor overlap")
    model = LinearRegression().fit(
        donor_mean.loc[valid].to_frame(),
        target.loc[valid],
    )
    test = donor_mean.reindex(predict_index)
    prediction = pd.Series(np.nan, index=predict_index, dtype=float)
    available = test.notna()
    if available.any():
        prediction.loc[available] = model.predict(
            test.loc[available].to_frame()
        )
    return prediction


def log_donor_regression_prediction(
    features: pd.DataFrame,
    target: pd.Series,
    predict_index: pd.DatetimeIndex,
) -> pd.Series:
    """Fit donor regression in log space so wind-speed predictions stay nonnegative."""
    donors = features.filter(regex=r"^donor__")
    if donors.empty:
        raise ValueError("no donor features")
    donor_mean = donors.mean(axis=1)
    valid = target.notna() & donor_mean.notna() & target.ge(0) & donor_mean.ge(0)
    if int(valid.sum()) < 200:
        raise ValueError("insufficient donor overlap")
    model = LinearRegression().fit(
        np.log1p(donor_mean.loc[valid]).to_frame(),
        np.log1p(target.loc[valid]),
    )
    test = donor_mean.reindex(predict_index)
    prediction = pd.Series(np.nan, index=predict_index, dtype=float)
    available = test.notna() & test.ge(0)
    if available.any():
        prediction.loc[available] = np.expm1(
            model.predict(np.log1p(test.loc[available]).to_frame())
        )
    return prediction.clip(lower=0.0)


def bounded_rh_donor_prediction(
    features: pd.DataFrame,
    target: pd.Series,
    predict_index: pd.DatetimeIndex,
) -> pd.Series:
    """Fit RH donor regression in logit space to keep predictions in 0--100%."""
    donors = features.filter(regex=r"^donor__")
    if donors.empty:
        raise ValueError("no donor features")
    donor_mean = donors.mean(axis=1)
    valid = target.notna() & donor_mean.notna()
    if int(valid.sum()) < 200:
        raise ValueError("insufficient donor overlap")

    epsilon = 0.1

    def logit(values: pd.Series) -> pd.Series:
        proportion = values.clip(epsilon, 100.0 - epsilon) / 100.0
        return np.log(proportion / (1.0 - proportion))

    model = LinearRegression().fit(
        logit(donor_mean.loc[valid]).to_frame(),
        logit(target.loc[valid]),
    )
    test = donor_mean.reindex(predict_index)
    prediction = pd.Series(np.nan, index=predict_index, dtype=float)
    available = test.notna()
    if available.any():
        transformed = model.predict(logit(test.loc[available]).to_frame())
        prediction.loc[available] = 100.0 / (1.0 + np.exp(-transformed))
    return prediction


def linear_run_prediction(
    target: pd.Series,
    run: pd.DatetimeIndex,
    parameter: str,
) -> pd.Series:
    method = "wind_angle" if parameter == "Wind direction" else "time"
    return interpolate_internal_run(target, run, method)


def fill_sparse_prediction_holes(
    prediction: pd.Series,
    parameter: str,
    limit: int = 24,
) -> pd.Series:
    """Interpolate short holes inside an otherwise available model prediction."""
    if not prediction.isna().any():
        return prediction
    result = prediction.copy()
    for run in contiguous_nan_runs(prediction):
        if len(run) <= limit and is_internal(prediction, run):
            result.loc[run] = linear_run_prediction(
                prediction,
                run,
                parameter,
            )
    return result


def boundary_blend_prediction(
    target: pd.Series,
    run: pd.DatetimeIndex,
    prediction: pd.Series,
) -> pd.Series:
    """Preserve a model's shape while matching a linear boundary connection."""
    if prediction.isna().any():
        return prediction
    linear = linear_run_prediction(target, run, "Tair")
    start_correction = float(linear.iloc[0] - prediction.iloc[0])
    end_correction = float(linear.iloc[-1] - prediction.iloc[-1])
    correction = np.linspace(start_correction, end_correction, len(run))
    return prediction + correction


def edge_blend_prediction(
    target: pd.Series,
    run: pd.DatetimeIndex,
    prediction: pd.Series,
    edge_hours: int = 24,
) -> pd.Series:
    """Taper only the edges of a long prediction toward observed boundaries."""
    if prediction.isna().any() or len(run) <= 2 * edge_hours:
        return boundary_blend_prediction(target, run, prediction)
    linear = linear_run_prediction(target, run, "Tair")
    result = prediction.copy()
    width = min(edge_hours, len(run))
    weights = np.linspace(1.0, 0.0, width)
    left_correction = float(linear.iloc[0] - result.iloc[0])
    right_correction = float(linear.iloc[-1] - result.iloc[-1])
    result.iloc[:width] += left_correction * weights
    result.iloc[-width:] += right_correction * weights[::-1]
    return result


def prediction_diagnostics(
    source: pd.Series,
    run: pd.DatetimeIndex,
    raw_prediction: pd.Series,
    parameter: str,
) -> Tuple[pd.Series, int, float, float, float, List[str]]:
    prediction = apply_bounds(raw_prediction, parameter)
    lower, upper = PHYSICAL_BOUNDS[parameter]
    violations = int((raw_prediction < lower).sum())
    if upper is not None:
        violations += int((raw_prediction > upper).sum())

    left = run[0] - pd.Timedelta(hours=1)
    right = run[-1] + pd.Timedelta(hours=1)
    if prediction.notna().all():
        left_jump = boundary_jump(
            parameter,
            float(source.loc[left]),
            float(prediction.iloc[0]),
        )
        right_jump = boundary_jump(
            parameter,
            float(prediction.iloc[-1]),
            float(source.loc[right]),
        )
    else:
        left_jump = np.nan
        right_jump = np.nan

    if parameter == "Wind direction":
        internal_jumps = (
            (prediction.diff() + 180.0) % 360.0 - 180.0
        ).abs()
    else:
        internal_jumps = prediction.diff().abs()
    max_internal_jump = float(internal_jumps.max()) if len(prediction) > 1 else 0.0

    qc_reasons = []
    if violations:
        qc_reasons.append("physical_clip")
    max_jump = max(left_jump, right_jump)
    if np.isfinite(max_jump) and max_jump > BOUNDARY_REVIEW_LIMITS[parameter]:
        qc_reasons.append("boundary_jump")
    if (
        np.isfinite(max_internal_jump)
        and max_internal_jump > HOURLY_JUMP_REVIEW_LIMITS[parameter]
    ):
        qc_reasons.append("internal_hourly_jump")
    return (
        prediction,
        violations,
        left_jump,
        right_jump,
        max_internal_jump,
        qc_reasons,
    )


def repair_review_prediction(
    parameter: str,
    category: str,
    method: str,
    source: pd.Series,
    run: pd.DatetimeIndex,
    raw_prediction: pd.Series,
    features: pd.DataFrame,
    qc_reasons: List[str],
) -> Tuple[pd.Series, str, str]:
    """Apply a narrow, parameter-specific repair only to a flagged prediction."""
    if not qc_reasons:
        return raw_prediction, "", ""
    try:
        if parameter == "Srad" and category == "short" and "boundary_jump" in qc_reasons:
            return linear_run_prediction(source, run, parameter), "short_linear", ""
        if (
            parameter == "Wind direction"
            and category in {"short", "medium"}
            and "boundary_jump" in qc_reasons
        ):
            return (
                linear_run_prediction(source, run, parameter),
                "circular_interpolation",
                "",
            )
        if (
            parameter == "Wind speed"
            and method == "donor_regression"
            and "physical_clip" in qc_reasons
        ):
            return (
                log_donor_regression_prediction(features, source, run),
                "log_donor_regression",
                "",
            )
        if parameter == "RH":
            repaired = raw_prediction.copy()
            methods = []
            if "physical_clip" in qc_reasons:
                if method == "donor_regression":
                    repaired = bounded_rh_donor_prediction(
                        features,
                        source,
                        run,
                    )
                    methods.append("bounded_rh_donor")
                else:
                    repaired = apply_bounds(repaired, parameter)
                    methods.append("bounded_clip")
            if "boundary_jump" in qc_reasons:
                repaired = edge_blend_prediction(source, run, repaired)
                methods.append("edge_blend")
            return repaired.clip(0.0, 100.0), "+".join(methods), ""
        if parameter == "Tair" and "boundary_jump" in qc_reasons:
            if category == "short":
                repaired = boundary_blend_prediction(
                    source,
                    run,
                    raw_prediction,
                )
                return repaired, "boundary_blend", ""
            repaired = edge_blend_prediction(source, run, raw_prediction)
            return repaired, "edge_blend", ""
        if "physical_clip" in qc_reasons:
            return apply_bounds(raw_prediction, parameter), "bounded_clip", ""
    except Exception as exc:
        return raw_prediction, "", str(exc)
    return raw_prediction, "", ""


def method_map_frame() -> pd.DataFrame:
    rows = []
    for parameter, gap_methods in MET_GAP_METHODS.items():
        for category, method in gap_methods.items():
            rows.append(
                {
                    "Parameter": parameter,
                    "Gap Class": category,
                    "Method": method,
                    "Status": "expanded_benchmark_selection",
                }
            )
    return pd.DataFrame(rows)


def fill_full_station(
    station: str,
    frame: pd.DataFrame,
    source_frames: Dict[str, pd.DataFrame],
    parameters: Iterable[str],
    repair_review: bool = False,
    sensor_qc_mask_counts: Dict[Tuple[str, str], int] | None = None,
) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    available = [parameter for parameter in ALL_MET_PARAMS if parameter in frame]
    result = frame[available].copy()
    detail_rows: List[dict] = []
    summary_rows: List[dict] = []
    rng = np.random.default_rng(MODEL_RANDOM_SEED)

    for parameter in parameters:
        if parameter not in frame.columns:
            summary_rows.append(
                {
                    "Station": station,
                    "Parameter": parameter,
                    "Status": "column_missing",
                    "Total Hours": len(frame),
                }
            )
            continue

        source = pd.to_numeric(frame[parameter], errors="coerce")
        filled = source.copy()
        internal_runs = [
            run for run in contiguous_nan_runs(source) if is_internal(source, run)
        ]
        features = build_model_features(
            station, parameter, source, source_frames
        )
        predictions_by_method: Dict[str, pd.Series] = {}
        errors_by_method: Dict[str, str] = {}

        for method in sorted(
            {
                MET_GAP_METHODS[parameter][gap_class(len(run))]
                for run in internal_runs
            }
        ):
            method_runs = [
                run
                for run in internal_runs
                if MET_GAP_METHODS[parameter][gap_class(len(run))] == method
            ]
            predict_index = pd.DatetimeIndex(
                sorted(timestamp for run in method_runs for timestamp in run)
            )
            try:
                if method == "donor_regression":
                    predictions_by_method[method] = donor_regression_prediction(
                        features, source, predict_index
                    )
                elif method in {"random_forest", "xgboost"}:
                    predictions_by_method[method] = tree_prediction(
                        method,
                        features,
                        source,
                        predict_index,
                        parameter,
                        rng,
                    )
                elif method == "linear":
                    predictions_by_method[method] = pd.Series(
                        np.nan, index=predict_index, dtype=float
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
            except Exception as exc:
                errors_by_method[method] = str(exc)

        for run in internal_runs:
            category = gap_class(len(run))
            method = MET_GAP_METHODS[parameter][category]
            fallback = ""
            error = errors_by_method.get(method, "")
            if method == "linear":
                prediction = linear_run_prediction(source, run, parameter)
            elif error:
                prediction = pd.Series(np.nan, index=run, dtype=float)
            else:
                prediction = predictions_by_method[method].reindex(run)

            if method == "donor_regression" and prediction.isna().any():
                repaired_prediction = fill_sparse_prediction_holes(
                    prediction,
                    parameter,
                )
                if repaired_prediction.notna().sum() > prediction.notna().sum():
                    prediction = repaired_prediction
                    fallback = "prediction_interpolation_<24h"

            missing = prediction.isna()
            if missing.any() and category == "short":
                linear = linear_run_prediction(source, run, parameter)
                prediction.loc[missing] = linear.loc[missing]
                if prediction.notna().any():
                    fallback = ";".join(
                        value for value in [fallback, "linear"] if value
                    )

            raw_prediction = prediction.copy()
            (
                prediction,
                violations,
                left_jump,
                right_jump,
                max_internal_jump,
                qc_reasons,
            ) = prediction_diagnostics(
                source,
                run,
                raw_prediction,
                parameter,
            )
            original_qc_reason = ";".join(qc_reasons)
            repair_method = ""
            repair_error = ""
            if repair_review and qc_reasons:
                repaired_raw, repair_method, repair_error = (
                    repair_review_prediction(
                        parameter,
                        category,
                        method,
                        source,
                        run,
                        raw_prediction,
                        features,
                        qc_reasons,
                    )
                )
                if repair_method:
                    raw_prediction = repaired_raw
                    (
                        prediction,
                        violations,
                        left_jump,
                        right_jump,
                        max_internal_jump,
                        qc_reasons,
                    ) = prediction_diagnostics(
                        source,
                        run,
                        raw_prediction,
                        parameter,
                    )

            valid = prediction.notna()
            filled.loc[run[valid.to_numpy()]] = prediction.loc[valid].to_numpy()
            filled_hours = int(valid.sum())
            remaining_hours = len(run) - filled_hours
            action = (
                "filled"
                if remaining_hours == 0
                else "partially_filled"
                if filled_hours
                else "failed"
            )
            if remaining_hours:
                qc_reasons.append("remaining_prediction_nan")

            qc_status = "review" if qc_reasons else "accepted"
            detail_rows.append(
                {
                    "Station": station,
                    "Parameter": parameter,
                    "Gap Class": category,
                    "Start": run[0],
                    "End": run[-1],
                    "Hours": len(run),
                    "Method": method,
                    "Fallback": fallback,
                    "Action": action,
                    "Filled Hours": filled_hours,
                    "Remaining Hours": remaining_hours,
                    "Raw Physical Violations": violations,
                    "Left Boundary Jump": left_jump,
                    "Right Boundary Jump": right_jump,
                    "Max Internal Hourly Jump": max_internal_jump,
                    "Original QC Reason": original_qc_reason,
                    "Repair Method": repair_method,
                    "QC Status": qc_status,
                    "QC Reason": ";".join(qc_reasons),
                    "Error": ";".join(
                        value for value in [error, repair_error] if value
                    ),
                }
            )

        result[parameter] = filled
        parameter_details = [
            row for row in detail_rows if row["Parameter"] == parameter
        ]
        summary_rows.append(
            {
                "Station": station,
                "Parameter": parameter,
                "Status": "ok",
                "Total Hours": len(source),
                "Observed Source Hours": int(source.notna().sum()),
                "Source NaN Hours": int(source.isna().sum()),
                "Internal Gap Segments": len(internal_runs),
                "Internal Gap Hours": sum(len(run) for run in internal_runs),
                "Model Filled Hours": sum(
                    row["Filled Hours"] for row in parameter_details
                ),
                "Failed Segments": sum(
                    row["Action"] != "filled" for row in parameter_details
                ),
                "Review Segments": sum(
                    row["QC Status"] == "review"
                    for row in parameter_details
                ),
                "Repaired Segments": sum(
                    bool(row["Repair Method"])
                    for row in parameter_details
                ),
                "Sensor QC Masked Hours": int(
                    (sensor_qc_mask_counts or {}).get(
                        (station, parameter),
                        0,
                    )
                ),
                "Remaining NaN Hours": int(filled.isna().sum()),
            }
        )

    result.index.name = "Date"
    return result, summary_rows, detail_rows


def met_delivery_base(station: str, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Load the most complete existing MET frame without touching soil outputs."""
    candidates = [
        DEFAULT_OUTPUT_DIR / f"Station{station}_met_filled_allgaps.csv",
        DEFAULT_OUTPUT_DIR / f"Station{station}_met_filled_shortgaps.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
        frame.index = pd.DatetimeIndex(frame.index)
        frame.index.name = "Date"
        frame = frame[~frame.index.duplicated(keep="first")].sort_index()
        return frame.reindex(cleaned.index)

    available = [parameter for parameter in ALL_MET_PARAMS if parameter in cleaned]
    return cleaned[available].copy()


def load_reconciled_ppt_sources(
    args: argparse.Namespace,
) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    """Load all station Ppt series after applying the approved source order."""
    frames: Dict[str, pd.DataFrame] = {}
    comparisons: List[dict] = []
    for station in discover_stations():
        frame = read_cleaned_station(station)
        if "Ppt" not in frame:
            continue
        reconciled, comparison = reconcile_station_ppt(
            station,
            frame,
            args.soil_base_dir,
            args.met_base_dir,
        )
        frame = frame.copy()
        frame["Ppt"] = reconciled
        frames[station] = frame
        comparisons.append(comparison)
    return frames, comparisons


def fit_missing_ppt(
    station: str,
    source: pd.Series,
    source_frames: Dict[str, pd.DataFrame],
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series, str]:
    """Fit one station's missing Ppt and return predictions plus donor support."""
    missing_index = pd.DatetimeIndex(source.index[source.isna()])
    filled = source.copy()
    model_output = pd.DataFrame(
        index=missing_index,
        columns=["Prediction", "Rain Probability", "Rain Threshold"],
        dtype=float,
    )
    donor_counts = pd.Series(0, index=missing_index, dtype=int)
    donor_wet_counts = pd.Series(0, index=missing_index, dtype=int)
    if not len(missing_index):
        return filled, model_output, donor_counts, donor_wet_counts, ""

    try:
        station_seed = MODEL_RANDOM_SEED + sum(
            (position + 1) * ord(character)
            for position, character in enumerate(station)
        )
        features = build_ppt_features(station, source, source_frames)
        donor_counts = (
            features["donor_available_count"]
            .reindex(missing_index)
            .fillna(0)
            .astype(int)
        )
        donor_wet_counts = (
            features["donor_wet_count"]
            .reindex(missing_index)
            .fillna(0)
            .astype(int)
        )
        model_output = two_part_ppt_prediction(
            features,
            source,
            missing_index,
            np.random.default_rng(station_seed),
        )
        prediction = pd.to_numeric(model_output["Prediction"], errors="coerce")
        if prediction.isna().any() or (~np.isfinite(prediction)).any():
            raise ValueError("Ppt model returned missing or non-finite values")
        if prediction.lt(0.0).any():
            raise ValueError("Ppt model returned negative precipitation")
        filled.loc[missing_index] = prediction
        return filled, model_output, donor_counts, donor_wet_counts, ""
    except Exception as exc:
        return filled, model_output, donor_counts, donor_wet_counts, str(exc)


def ppt_probability_state(
    model_output: pd.DataFrame,
) -> Tuple[float, pd.Series, pd.Series, float, float]:
    thresholds = pd.to_numeric(
        model_output.get("Rain Threshold", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    threshold = (
        float(thresholds.iloc[0])
        if len(thresholds)
        else PPT_FALLBACK_RAIN_PROBABILITY_THRESHOLD
    )
    low = max(0.0, threshold - PPT_UNCERTAIN_THRESHOLD_MARGIN)
    high = min(1.0, threshold + PPT_UNCERTAIN_THRESHOLD_MARGIN)
    probability = pd.to_numeric(
        model_output.get("Rain Probability", pd.Series(dtype=float)),
        errors="coerce",
    )
    uncertain = probability.between(low, high, inclusive="both")
    return threshold, probability, uncertain, low, high


def ppt_segment_rows(
    station: str,
    source: pd.Series,
    filled: pd.Series,
    probability: pd.Series,
    donor_counts: pd.Series,
    donor_wet_counts: pd.Series,
    threshold: float,
    low: float,
    high: float,
    model_error: str,
) -> List[dict]:
    rows: List[dict] = []
    for run in contiguous_nan_runs(source):
        prediction = filled.reindex(run)
        run_probability = probability.reindex(run)
        run_donors = donor_counts.reindex(run).fillna(0).astype(int)
        run_wet_donors = donor_wet_counts.reindex(run).fillna(0).astype(int)
        remaining = int(prediction.isna().sum())
        zero_donor_hours = int(run_donors.eq(0).sum())
        regional_rain_hours = int(run_wet_donors.ge(3).sum())
        predicted_wet_hours = int(prediction.gt(0.0).sum())
        max_wet_donors = int(run_wet_donors.max())
        max_probability = float(run_probability.max())
        threshold_margin = threshold - max_probability

        if remaining:
            qc_status, qc_reason, priority = (
                "failed", "remaining_prediction_nan", "high"
            )
        elif zero_donor_hours:
            qc_status, qc_reason, priority = (
                "review", "no_concurrent_donor", "medium"
            )
        elif regional_rain_hours and predicted_wet_hours == 0:
            priority = (
                "high"
                if max_wet_donors >= 10 or threshold_margin <= 0.05
                else "medium"
            )
            qc_status, qc_reason = "review", "regional_rain_without_prediction"
        else:
            qc_status, qc_reason, priority = "accepted", "", ""

        rows.append(
            {
                "Station": station,
                "Gap Class": gap_class(len(run)),
                "Start": run[0],
                "End": run[-1],
                "Hours": len(run),
                "Internal Gap": is_internal(source, run),
                "Method": PPT_MODEL_NAME,
                "Rain Probability Threshold": threshold,
                "Action": "filled" if remaining == 0 else "failed",
                "Predicted Wet Hours": predicted_wet_hours,
                "Predicted Total Ppt": float(prediction.sum(min_count=1)),
                "Median Donor Count": float(run_donors.median()),
                "Three Plus Donor Wet Hours": regional_rain_hours,
                "Max Donor Wet Count": max_wet_donors,
                "Zero Donor Hours": zero_donor_hours,
                "Uncertain Occurrence Hours": int(
                    run_probability.between(low, high, inclusive="both").sum()
                ),
                "Max Rain Probability": max_probability,
                "Threshold Margin": threshold_margin,
                "Remaining Hours": remaining,
                "QC Status": qc_status,
                "QC Reason": qc_reason,
                "Review Priority": priority,
                "Error": model_error,
            }
        )
    return rows


def ppt_hourly_report(
    station: str,
    source: pd.Series,
    filled: pd.Series,
    probability: pd.Series,
    threshold: float,
    donor_counts: pd.Series,
    donor_wet_counts: pd.Series,
) -> pd.DataFrame:
    index = pd.DatetimeIndex(source.index[source.isna()])
    hourly = pd.DataFrame(index=index)
    hourly.index.name = "Date"
    hourly["Station"] = station
    hourly["Filled Ppt"] = filled.reindex(index)
    hourly["Rain Probability"] = probability.reindex(index)
    hourly["Rain Threshold"] = threshold
    hourly["Predicted Wet"] = hourly["Filled Ppt"].gt(0.0)
    hourly["Donor Observed Count"] = donor_counts.reindex(index)
    hourly["Donor Wet Count"] = donor_wet_counts.reindex(index)
    hourly["Support"] = np.select(
        [
            hourly["Donor Observed Count"].eq(0),
            hourly["Donor Observed Count"].lt(3),
        ],
        ["no_concurrent_donor", "one_or_two_donors"],
        default="three_or_more_donors",
    )
    hourly["Method"] = PPT_MODEL_NAME
    return hourly


def ppt_station_row(
    station: str,
    source: pd.Series,
    filled: pd.Series,
    threshold: float,
    uncertain: pd.Series,
    donor_counts: pd.Series,
    donor_wet_counts: pd.Series,
    segments: List[dict],
    model_error: str,
) -> dict:
    missing_index = pd.DatetimeIndex(source.index[source.isna()])
    observed = source.notna()
    observed_changed = int(
        (~np.isclose(
            source.loc[observed].to_numpy(dtype=float),
            filled.loc[observed].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        )).sum()
    )
    review_segments = sum(row["QC Status"] == "review" for row in segments)
    regional_reviews = sum(
        row["QC Reason"] == "regional_rain_without_prediction"
        for row in segments
    )
    status = "failed" if model_error else "review" if review_segments else "ok"
    return {
        "Station": station,
        "Status": status,
        "Total Hours": len(source),
        "Direct Source Hours": int(source.notna().sum()),
        "Source NaN Hours": int(source.isna().sum()),
        "Model Filled Hours": int(source.isna().sum() - filled.isna().sum()),
        "Predicted Wet Hours": int(filled.loc[missing_index].gt(0.0).sum()),
        "Predicted Ppt Total": (
            float(filled.loc[missing_index].sum()) if len(missing_index) else 0.0
        ),
        "Uncertain Occurrence Hours": int(uncertain.sum()),
        "Rain Probability Threshold": threshold,
        "Zero Donor Hours": int(donor_counts.eq(0).sum()),
        "Three Plus Donor Wet Hours": int(donor_wet_counts.ge(3).sum()),
        "Review Segments": review_segments,
        "Regional Rain Review Segments": regional_reviews,
        "Remaining NaN Hours": int(filled.isna().sum()),
        "Observed Hours Changed": observed_changed,
        "Model": PPT_MODEL_NAME,
        "Error": model_error,
    }


def run_ppt_stage(
    args: argparse.Namespace,
    requested_stations: List[str],
) -> None:
    """Fill Ppt missing from all approved direct sources for selected stations."""
    source_frames, comparison_rows = load_reconciled_ppt_sources(args)

    stations = [station for station in requested_stations if station in source_frames]
    missing_stations = sorted(set(requested_stations) - set(stations))
    if missing_stations:
        raise ValueError(
            "Selected stations have no Ppt source column: "
            + ", ".join(missing_stations)
        )

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = DEFAULT_REPORT_DIR / (
            "ppt_model_fill_targeted" if args.station else "ppt_model_fill"
        )
    hourly_dir = report_dir / "hourly"
    report_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir.mkdir(parents=True, exist_ok=True)
    if args.write:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    station_rows: List[dict] = []
    segment_rows: List[dict] = []

    for station in stations:
        frame = source_frames[station]
        source = pd.to_numeric(frame["Ppt"], errors="coerce")
        (
            filled,
            model_output,
            donor_counts,
            donor_wet_counts,
            model_error,
        ) = fit_missing_ppt(station, source, source_frames)
        rain_threshold, probability, uncertain, low, high = (
            ppt_probability_state(model_output)
        )

        station_segments = ppt_segment_rows(
            station,
            source,
            filled,
            probability,
            donor_counts,
            donor_wet_counts,
            rain_threshold,
            low,
            high,
            model_error,
        )
        segment_rows.extend(station_segments)

        hourly = ppt_hourly_report(
            station,
            source,
            filled,
            probability,
            rain_threshold,
            donor_counts,
            donor_wet_counts,
        )
        hourly.to_csv(
            hourly_dir / f"Station{station}_ppt_model_fill_hourly.csv",
            na_rep="NaN",
        )
        station_rows.append(
            ppt_station_row(
                station,
                source,
                filled,
                rain_threshold,
                uncertain,
                donor_counts,
                donor_wet_counts,
                station_segments,
                model_error,
            )
        )

        if args.write:
            delivery = met_delivery_base(station, frame)
            delivery["Ppt"] = filled
            columns = [parameter for parameter in ALL_MET_PARAMS if parameter in delivery]
            delivery[columns].to_csv(
                args.output_dir / f"Station{station}_met_filled_complete.csv",
                na_rep="NaN",
            )

    summary = pd.DataFrame(station_rows)
    details = pd.DataFrame(segment_rows, columns=PPT_MODEL_DETAIL_COLUMNS)
    summary.to_csv(report_dir / "ppt_model_fill_station_summary.csv", index=False)
    details.to_csv(report_dir / "ppt_model_fill_segment_detail.csv", index=False)
    review_queue = details.loc[details["QC Status"].eq("review")].copy()
    if not review_queue.empty:
        priority_rank = review_queue["Review Priority"].map(
            {"high": 0, "medium": 1}
        ).fillna(2)
        review_queue = (
            review_queue.assign(_priority_rank=priority_rank)
            .sort_values(
                [
                    "_priority_rank",
                    "Max Donor Wet Count",
                    "Threshold Margin",
                    "Station",
                    "Start",
                ],
                ascending=[True, False, True, True, True],
            )
            .drop(columns="_priority_rank")
        )
    review_queue.to_csv(
        report_dir / "ppt_model_fill_review_queue.csv",
        index=False,
    )
    pd.DataFrame(comparison_rows, columns=PPT_COMPARISON_COLUMNS).to_csv(
        report_dir / "ppt_source_comparison_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "Source Policy": "dedicated MET first; soil-file fallback",
                "Missing Source Policy": "retain NaN, then model impute",
                "Model": PPT_MODEL_NAME,
                "Rain Probability Threshold": "station OOB CSI; fallback 0.5",
                "Feature Set": (
                    "hour/day-of-year cycles + concurrent donor availability, "
                    "wet count, total, max, positive mean, positive median"
                ),
                "Maximum Training Rows": MAX_TRAIN_ROWS,
                "Random Seed": MODEL_RANDOM_SEED,
            }
        ]
    ).to_csv(report_dir / "ppt_model_configuration.csv", index=False)

    print("Ppt source reconciliation and two-part model filling complete.")
    print(f"Stations processed: {len(stations)}")
    print(
        "Ppt hours filled: "
        f"{int(summary['Model Filled Hours'].fillna(0).sum())}"
    )
    print(
        "Remaining Ppt NaN hours: "
        f"{int(summary['Remaining NaN Hours'].fillna(0).sum())}"
    )
    print(
        "Observed Ppt hours changed: "
        f"{int(summary['Observed Hours Changed'].fillna(0).sum())}"
    )
    print(f"Reports written under: {report_dir}")
    if args.write:
        print(f"Complete MET outputs written under: {args.output_dir}")


def run_full_stage(
    args: argparse.Namespace,
    requested_stations: List[str],
    parameters: List[str],
) -> None:
    source_stations = discover_met_source_stations()
    source_frames = {
        station: read_cleaned_station(station)
        for station in source_stations
    }

    ppt_rows = []
    for station, frame in source_frames.items():
        if "Ppt" not in frame:
            continue
        reconciled, comparison = reconcile_station_ppt(
            station,
            frame,
            args.soil_base_dir,
            args.met_base_dir,
        )
        frame["Ppt"] = reconciled
        ppt_rows.append(comparison)

    stations = [
        station for station in requested_stations if station in source_frames
    ]
    skipped = sorted(set(requested_stations) - set(stations))
    if skipped:
        print(
            "Skipping stations without dedicated non-precipitation MET data: "
            + ", ".join(skipped)
        )
    if not stations:
        raise ValueError("No selected station has non-precipitation MET data")

    selected = [
        parameter for parameter in parameters
        if parameter in NON_PPT_MET_PARAMS
    ]
    if not selected:
        raise ValueError(
            "Full MET filling currently supports Tair, RH, Srad, Wind speed, "
            "and Wind direction. Run --ppt-full for Ppt model filling."
        )
    if "Ppt" in parameters:
        print(
            "Ppt is reconciled here; its separate two-part model runs with "
            "--ppt-full (or the met-ppt runner stage)."
        )

    sensor_qc_rows: List[dict] = []
    sensor_qc_mask_counts: Dict[Tuple[str, str], int] = {}
    if args.repair_review:
        (
            source_frames,
            sensor_qc_rows,
            sensor_qc_mask_counts,
        ) = apply_network_sensor_qc(source_frames, selected)

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = DEFAULT_REPORT_DIR / (
            "model_fill_targeted" if args.station or args.param else "model_fill"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.write:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_details = []
    for station in stations:
        result, summary, details = fill_full_station(
            station,
            source_frames[station],
            source_frames,
            selected,
            repair_review=args.repair_review,
            sensor_qc_mask_counts=sensor_qc_mask_counts,
        )
        all_summary.extend(summary)
        all_details.extend(details)
        if args.write:
            output_path = (
                args.output_dir / f"Station{station}_met_filled_allgaps.csv"
            )
            if args.param:
                result = preserve_unselected_output(result, output_path, selected)
            result.to_csv(output_path, na_rep="NaN")
            if args.param:
                complete_path = (
                    args.output_dir
                    / f"Station{station}_met_filled_complete.csv"
                )
                if complete_path.exists():
                    complete = preserve_unselected_output(
                        result,
                        complete_path,
                        selected,
                    )
                    complete.to_csv(complete_path, na_rep="NaN")

    summary = pd.DataFrame(all_summary)
    details = pd.DataFrame(all_details, columns=MODEL_DETAIL_COLUMNS)
    summary.to_csv(
        report_dir / "met_model_fill_station_summary.csv", index=False
    )
    details.to_csv(
        report_dir / "met_model_fill_segment_detail.csv", index=False
    )
    selected_methods = method_map_frame()
    selected_methods.to_csv(
        report_dir / "met_selected_method_map.csv", index=False
    )
    # Keep one canonical map separate from historical full/targeted reports.
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected_methods.to_csv(
        DEFAULT_REPORT_DIR / "met_selected_method_map.csv", index=False
    )
    pd.DataFrame(ppt_rows).to_csv(
        report_dir / "ppt_source_comparison_summary.csv", index=False
    )
    if args.repair_review:
        pd.DataFrame(
            sensor_qc_rows,
            columns=SENSOR_QC_REPORT_COLUMNS,
        ).to_csv(
            report_dir / "met_sensor_qc_masked_segments.csv",
            index=False,
        )

    print("Full internal non-precipitation MET filling complete.")
    print(f"Stations processed: {len(stations)}")
    print(
        "Internal hours filled: "
        f"{int(summary['Model Filled Hours'].fillna(0).sum())}"
    )
    print(
        "Failed segments: "
        f"{int(summary['Failed Segments'].fillna(0).sum())}"
    )
    print(
        "Review segments: "
        f"{int(summary['Review Segments'].fillna(0).sum())}"
    )
    if args.repair_review:
        print(
            "Sensor-QC observations masked: "
            f"{int(summary['Sensor QC Masked Hours'].fillna(0).sum())}"
        )
        print(
            "QC-review segments repaired: "
            f"{int(summary['Repaired Segments'].fillna(0).sum())}"
        )
    print(f"Reports written under: {report_dir}")
    if args.write:
        print(f"MET all-gap outputs written under: {args.output_dir}")


def summarize_and_fill(
    station: str,
    df: pd.DataFrame,
    parameters: Iterable[str],
    max_gap: int,
) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    available = [parameter for parameter in parameters if parameter in df.columns]
    result = df[available].copy()
    summary_rows: List[dict] = []
    gap_rows: List[dict] = []

    for parameter in parameters:
        if parameter not in df.columns:
            summary_rows.append(
                {
                    "Station": station,
                    "Parameter": parameter,
                    "Status": "column_missing",
                    "Total Hours": len(df),
                }
            )
            continue

        source = pd.to_numeric(df[parameter], errors="coerce")
        filled = source.copy()
        observed = source.dropna()
        short_filled = 0
        internal_nan_hours = 0
        outside_coverage_hours = 0

        for run in contiguous_nan_runs(source):
            hours = len(run)
            internal = is_internal(source, run)
            action = "left_unfilled"
            method = "none"

            if not internal:
                outside_coverage_hours += hours
                action = "outside_observed_coverage"
            else:
                internal_nan_hours += hours
                if hours < max_gap and parameter == "Ppt":
                    action = "deferred_precipitation"
                elif hours < max_gap:
                    method = short_interp_for(parameter)
                    predictions = apply_bounds(interpolate_internal_run(filled, run, method), parameter)
                    valid = predictions.notna()
                    if valid.all():
                        filled.loc[run] = predictions.values
                        short_filled += hours
                        action = "filled"
                    else:
                        action = "interpolation_failed"

            gap_rows.append(
                {
                    "Station": station,
                    "Parameter": parameter,
                    "Start": run[0],
                    "End": run[-1],
                    "Hours": hours,
                    "Category": gap_category(hours),
                    "Internal Gap": internal,
                    "Action": action,
                    "Method": method,
                }
            )

        result[parameter] = filled
        summary_rows.append(
            {
                "Station": station,
                "Parameter": parameter,
                "Status": "ok" if len(observed) else "no_source_observations",
                "Total Hours": len(source),
                "Observed Source Hours": int(source.notna().sum()),
                "Source NaN Hours": int(source.isna().sum()),
                "Internal NaN Hours": internal_nan_hours,
                "Outside Coverage Hours": outside_coverage_hours,
                "Short Hours Filled": short_filled,
                "Remaining NaN Hours": int(filled.isna().sum()),
                "Observed Start": observed.index.min() if len(observed) else pd.NaT,
                "Observed End": observed.index.max() if len(observed) else pd.NaT,
            }
        )

    result.index.name = "Date"
    return result, summary_rows, gap_rows


def main() -> None:
    args = parse_args()
    stations = args.station if args.station else discover_stations()
    parameters = args.param if args.param else ALL_MET_PARAMS
    unknown = sorted(set(parameters) - set(ALL_MET_PARAMS))
    if unknown:
        raise ValueError(f"Unsupported MET parameter(s): {', '.join(unknown)}")
    if args.max_gap <= 1:
        raise ValueError("--max-gap must be greater than 1")
    if args.repair_review and not args.full:
        raise ValueError("--repair-review requires --full")
    if args.full and args.ppt_full:
        raise ValueError("--full and --ppt-full are separate stages")
    if args.ppt_full and args.param and set(args.param) != {"Ppt"}:
        raise ValueError("--ppt-full supports only the Ppt parameter")
    needs_raw_ppt = "Ppt" in parameters or args.full
    if needs_raw_ppt and not args.soil_base_dir.is_dir():
        raise FileNotFoundError(f"Raw soil directory not found: {args.soil_base_dir}")
    if needs_raw_ppt and not args.met_base_dir.is_dir():
        raise FileNotFoundError(f"Raw MET directory not found: {args.met_base_dir}")
    if args.ppt_full:
        run_ppt_stage(args, stations)
        return
    if args.full:
        run_full_stage(args, stations, parameters)
        return

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = DEFAULT_REPORT_DIR / "targeted" if args.station else DEFAULT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.write:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summary: List[dict] = []
    all_gaps: List[dict] = []
    ppt_comparison_rows: List[dict] = []
    for station in stations:
        df = read_cleaned_station(station)
        ppt_comparison = None
        if "Ppt" in parameters and "Ppt" in df.columns:
            reconciled_ppt, ppt_comparison = reconcile_station_ppt(
                station,
                df,
                args.soil_base_dir,
                args.met_base_dir,
            )
            df = df.copy()
            df["Ppt"] = reconciled_ppt
            ppt_comparison_rows.append(ppt_comparison)

        result, summary_rows, gap_rows = summarize_and_fill(station, df, parameters, args.max_gap)
        if ppt_comparison is not None:
            for row in summary_rows:
                if row["Parameter"] == "Ppt":
                    row["Ppt Source Recovered Hours"] = ppt_comparison.get(
                        "Additional Hours Recovered",
                        0,
                    )
        all_summary.extend(summary_rows)
        all_gaps.extend(gap_rows)
        if args.write:
            output_path = (
                args.output_dir
                / f"Station{station}_met_filled_shortgaps.csv"
            )
            if args.param:
                result = preserve_unselected_output(
                    result,
                    output_path,
                    parameters,
                )
            result.to_csv(output_path, na_rep="NaN")

    summary = pd.DataFrame(all_summary)
    gaps = pd.DataFrame(
        all_gaps,
        columns=[
            "Station", "Parameter", "Start", "End", "Hours", "Category",
            "Internal Gap", "Action", "Method",
        ],
    )
    ppt_comparison = pd.DataFrame(
        ppt_comparison_rows,
        columns=PPT_COMPARISON_COLUMNS,
    )
    summary.to_csv(report_dir / "met_station_parameter_summary.csv", index=False)
    gaps.to_csv(report_dir / "met_gap_inventory.csv", index=False)
    ppt_comparison.to_csv(report_dir / "ppt_source_comparison_summary.csv", index=False)

    filled_hours = int(summary.get("Short Hours Filled", pd.Series(dtype=float)).fillna(0).sum())
    deferred_ppt = int((gaps.get("Action", pd.Series(dtype=str)) == "deferred_precipitation").sum())
    recovered_ppt = int(
        ppt_comparison.get("Additional Hours Recovered", pd.Series(dtype=float)).fillna(0).sum()
    )
    print("MET audit and short-gap stage complete.")
    print(f"Stations audited: {len(stations)}")
    print(f"Short MET hours filled: {filled_hours}")
    print(f"Precipitation gaps deferred: {deferred_ppt}")
    print(f"Precipitation hours recovered from source reconciliation: {recovered_ppt}")
    print(f"Reports written under: {report_dir}")
    if args.write:
        print(f"MET-only outputs written under: {args.output_dir}")


if __name__ == "__main__":
    main()

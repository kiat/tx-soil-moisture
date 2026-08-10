"""Clear-sky-index and peer-station imputation for hourly solar radiation.

The program consumes the six anomaly-marked files created by
``generate_anomaly_report.py``.  It never overwrites ``Srad``: the observed
value is copied to ``Srad_original`` and all decisions are written to
``Srad_filled`` plus explicit audit columns.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astral import Observer
from astral.sun import sun


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "prewashed_anomaly_outputs" / "marked_data"
RAW_DIR = ROOT / "prewashed_met_data"
OUTPUT_DIR = ROOT / "imputed_solar_outputs"
FILLED_DIR = OUTPUT_DIR / "filled_data"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATE_COL = "Date"
SOLAR_COL = "Srad"
RSO_COL = "Rso"
TIMEZONE = "America/Chicago"
RANDOM_SEED = 20260728
RSO_MJ_TO_WM2 = 277.7778

STATION_LOCATIONS = {
    "CB01": {"latitude": 30.4193, "longitude": -98.8046},
    "CB04": {"latitude": 30.4600, "longitude": -98.9407},
    "CB06": {"latitude": 30.4421, "longitude": -98.8427},
    "FD02": {"latitude": 30.2456, "longitude": -98.6988},
    "FD03": {"latitude": 30.4175, "longitude": -98.8542},
    "WC05": {"latitude": 30.4319, "longitude": -98.8133},
}

FLAG_COLUMNS = [
    "night_radiation_anomaly_flag",
    "weather_related_low_radiation_event_flag",
    "long_zero_run_flag",
    "sudden_drop_flag",
    "missing_srad_flag",
    "sudden_spike_flag",
    "out_of_range_flag",
]
CORRECTION_FLAGS = [
    "night_radiation_anomaly_flag",
    "long_zero_run_flag",
    "sudden_drop_flag",
    "missing_srad_flag",
    "sudden_spike_flag",
    "out_of_range_flag",
]
PEER_REVIEW_FLAGS = ["long_zero_run_flag", "sudden_drop_flag", "sudden_spike_flag"]
OUTPUT_AUDIT_COLUMNS = [
    "Srad_original",
    "Srad_filled",
    "Srad_was_imputed",
    "Srad_was_corrected",
    "Srad_imputation_reason",
    "Srad_imputation_method",
    "Srad_imputation_sources",
    "Srad_peer_count",
    "Srad_imputation_uncertainty",
    "Srad_review_status",
]


@dataclass(frozen=True)
class Config:
    physical_min: float = 0.0
    physical_max: float = 1300.0
    daylight_buffer_minutes: int = 30
    min_rso_wm2: float = 20.0
    max_kt: float = 2.5
    min_calibration_pairs: int = 40
    max_calibration_pairs: int = 12000
    max_temporal_gap_hours: int = 3
    min_climatology_records: int = 20
    peer_consistency_abs_wm2: float = 80.0
    peer_consistency_relative: float = 0.30
    obvious_deviation_abs_wm2: float = 100.0
    obvious_deviation_relative: float = 0.40
    validation_segments_per_bucket: int = 10


def station_id(path: Path) -> str:
    """Extract the station identifier from a standard input filename."""
    return path.name.split("_")[0]


def as_bool(series: pd.Series) -> pd.Series:
    """Normalize mixed boolean representations into a clean Boolean series."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def robust_scale(values: np.ndarray) -> float:
    """Estimate robust variability from the median absolute deviation."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(max(1.4826 * mad, 1e-6))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the median of values after accounting for their weights."""
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = weights.sum() / 2
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def load_inputs(config: Config) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load and validate all station files, convert Rso units, and add daylight data.
    Return station datasets together with an input-quality summary."""
    files = sorted(INPUT_DIR.glob("*_met_anomaly_marked.csv"))
    found = {station_id(path) for path in files}
    expected = set(STATION_LOCATIONS)
    if found != expected:
        raise ValueError(f"Expected marked files for {sorted(expected)}, found {sorted(found)}")

    datasets: dict[str, pd.DataFrame] = {}
    checks: list[dict] = []
    for path in files:
        station = station_id(path)
        df = pd.read_csv(path, low_memory=False)
        required = {DATE_COL, SOLAR_COL, RSO_COL, *FLAG_COLUMNS}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        if df[DATE_COL].isna().any():
            raise ValueError(f"{path.name} contains invalid timestamps")
        if df[DATE_COL].duplicated().any() or not df[DATE_COL].is_monotonic_increasing:
            raise ValueError(f"{path.name} timestamps must be unique and sorted")
        for column in [SOLAR_COL, RSO_COL, "Ppt", "RH"]:
            if column in df:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        for column in FLAG_COLUMNS:
            df[column] = as_bool(df[column])

        raw_path = RAW_DIR / f"{station}_met.csv"
        raw = pd.read_csv(raw_path, usecols=[DATE_COL, SOLAR_COL], low_memory=False)
        raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
        timestamp_match = len(raw) == len(df) and raw[DATE_COL].equals(df[DATE_COL])
        srad_match = len(raw) == len(df) and np.allclose(
            pd.to_numeric(raw[SOLAR_COL], errors="coerce"),
            df[SOLAR_COL],
            equal_nan=True,
        )
        if not timestamp_match or not srad_match:
            raise ValueError(f"{path.name} does not faithfully match {raw_path.name}")

        finite_rso = df[RSO_COL].dropna()
        rso_max = float(finite_rso.max()) if not finite_rso.empty else np.nan
        if not np.isfinite(rso_max):
            unit = "unknown"
            factor = np.nan
        elif rso_max < 20:
            unit = "MJ m-2 h-1"
            factor = RSO_MJ_TO_WM2
        elif rso_max < 2000:
            unit = "W m-2"
            factor = 1.0
        else:
            raise ValueError(f"Implausible Rso range in {path.name}: maximum={rso_max:g}")
        df["Rso_wm2"] = df[RSO_COL] * factor
        df = add_daylight(df, station, config)
        datasets[station] = df
        checks.append(
            {
                "station": station,
                "rows": len(df),
                "start": df[DATE_COL].min(),
                "end": df[DATE_COL].max(),
                "median_interval_hours": df[DATE_COL].diff().median().total_seconds() / 3600,
                "duplicate_timestamps": int(df[DATE_COL].duplicated().sum()),
                "missing_Srad": int(df[SOLAR_COL].isna().sum()),
                "Rso_min": float(finite_rso.min()) if not finite_rso.empty else np.nan,
                "Rso_max": rso_max,
                "Rso_detected_unit": unit,
                "Rso_to_Wm2_factor": factor,
                "raw_timestamp_match": timestamp_match,
                "raw_Srad_match": srad_match,
            }
        )
    return datasets, pd.DataFrame(checks)


def add_daylight(df: pd.DataFrame, station: str, config: Config) -> pd.DataFrame:
    """Calculate station-specific sunrise, sunset, and daytime status for each row."""
    out = df.copy()
    location = STATION_LOCATIONS[station]
    observer = Observer(latitude=location["latitude"], longitude=location["longitude"])
    timezone = ZoneInfo(TIMEZONE)
    dates = out[DATE_COL].dt.date
    sunrise: dict = {}
    sunset: dict = {}
    for date in dates.drop_duplicates():
        solar = sun(observer, date=date, tzinfo=timezone)
        sunrise[date] = solar["sunrise"].replace(tzinfo=None)
        sunset[date] = solar["sunset"].replace(tzinfo=None)
    buffer = pd.Timedelta(minutes=config.daylight_buffer_minutes)
    out["_sunrise"] = pd.to_datetime(dates.map(sunrise))
    out["_sunset"] = pd.to_datetime(dates.map(sunset))
    out["_is_daytime"] = (out[DATE_COL] >= out["_sunrise"] - buffer) & (
        out[DATE_COL] <= out["_sunset"] + buffer
    )
    return out


def add_model_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add clear-sky index values and eligibility masks used by modeling and validation."""
    out = df.copy()
    any_flag = pd.Series(False, index=out.index)
    correction_flag = pd.Series(False, index=out.index)
    for column in FLAG_COLUMNS:
        any_flag |= out[column]
    for column in CORRECTION_FLAGS:
        correction_flag |= out[column]
    physically_valid = out[SOLAR_COL].between(config.physical_min, config.physical_max)
    rso_valid = out["Rso_wm2"].ge(config.min_rso_wm2)
    out["_normal"] = (
        ~any_flag
        & out[SOLAR_COL].notna()
        & physically_valid
        & out["_is_daytime"]
        & rso_valid
    )
    # Validation masks continuous unflagged, physically valid observations,
    # including normal nights. Rso availability is reflected in each method's
    # prediction coverage rather than used to cherry-pick validation segments.
    out["_validation_normal"] = (
        ~any_flag
        & out[SOLAR_COL].notna()
        & physically_valid
    )
    out["_peer_usable"] = (
        ~correction_flag
        & out[SOLAR_COL].notna()
        & physically_valid
        & out["_is_daytime"]
        & rso_valid
    )
    out["_kt"] = (out[SOLAR_COL] / out["Rso_wm2"]).where(rso_valid).clip(0, config.max_kt)
    return out


def aligned_frames(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Create timestamp-indexed station frames with all modeling columns attached."""
    return {station: add_model_columns(df, Config()).set_index(DATE_COL) for station, df in datasets.items()}


def robust_affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """Fit a robust affine relationship and return intercept, slope, scatter, and sample size."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2 or np.nanstd(x) < 1e-6:
        bias = float(np.nanmedian(y - x)) if n else 0.0
        residual = y - (x + bias)
        return bias, 1.0, robust_scale(residual), n
    design = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    weights = np.ones(n)
    for _ in range(8):
        residual = y - design @ beta
        scale = robust_scale(residual)
        if not np.isfinite(scale) or scale <= 1e-6:
            break
        cutoff = 1.5 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-9))
        beta = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    residual = y - design @ beta
    return float(beta[0]), float(np.clip(beta[1], 0.2, 2.5)), robust_scale(residual), n


def build_calibrations(
    aligned: dict[str, pd.DataFrame],
    config: Config,
    excluded: dict[str, set[pd.Timestamp]] | None = None,
) -> dict[tuple[str, str, int | str], dict]:
    """Fit monthly and all-season clear-sky-index relationships for every station pair."""
    excluded = excluded or {}
    result: dict[tuple[str, str, int | str], dict] = {}
    rng = np.random.default_rng(RANDOM_SEED)
    for target, target_df in aligned.items():
        target_ok = target_df["_normal"].copy()
        if excluded.get(target):
            target_ok.loc[target_ok.index.isin(excluded[target])] = False
        for peer, peer_df in aligned.items():
            if peer == target:
                continue
            joined = pd.DataFrame(
                {
                    "target": target_df["_kt"].where(target_ok),
                    "peer": peer_df["_kt"].where(peer_df["_normal"]),
                }
            ).dropna()
            if len(joined) >= config.min_calibration_pairs:
                sample = joined
                if len(sample) > config.max_calibration_pairs:
                    sample = sample.iloc[
                        np.sort(rng.choice(len(sample), config.max_calibration_pairs, replace=False))
                    ]
                intercept, slope, scatter, n = robust_affine_fit(
                    sample["peer"].to_numpy(), sample["target"].to_numpy()
                )
                result[(target, peer, "all")] = {
                    "intercept": intercept,
                    "slope": slope,
                    "scatter_kt": scatter,
                    "n": n,
                }
            for month in range(1, 13):
                month_data = joined[joined.index.month == month]
                if len(month_data) < config.min_calibration_pairs:
                    continue
                if len(month_data) > config.max_calibration_pairs:
                    month_data = month_data.iloc[
                        np.sort(rng.choice(len(month_data), config.max_calibration_pairs, replace=False))
                    ]
                intercept, slope, scatter, n = robust_affine_fit(
                    month_data["peer"].to_numpy(), month_data["target"].to_numpy()
                )
                result[(target, peer, month)] = {
                    "intercept": intercept,
                    "slope": slope,
                    "scatter_kt": scatter,
                    "n": n,
                }
    return result


def build_climatology(
    aligned: dict[str, pd.DataFrame],
    config: Config,
    excluded: dict[str, set[pd.Timestamp]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build station-specific month-hour clear-sky-index climatologies from normal data."""
    excluded = excluded or {}
    result = {}
    for station, df in aligned.items():
        ok = df["_normal"].copy()
        if excluded.get(station):
            ok.loc[ok.index.isin(excluded[station])] = False
        source = pd.DataFrame(
            {"month": df.index.month, "hour": df.index.hour, "kt": df["_kt"].where(ok)}
        ).dropna()
        grouped = source.groupby(["month", "hour"])["kt"].agg(["median", "count"])
        grouped["scatter"] = source.groupby(["month", "hour"])["kt"].apply(
            lambda values: robust_scale(values.to_numpy())
        )
        result[station] = grouped
    return result


def peer_predictions(
    station: str,
    timestamp: pd.Timestamp,
    target_rso: float,
    aligned: dict[str, pd.DataFrame],
    calibrations: dict[tuple[str, str, int | str], dict],
    config: Config,
) -> list[dict]:
    """Generate calibrated target-station Srad predictions from usable peer observations."""
    if not np.isfinite(target_rso) or target_rso < config.min_rso_wm2:
        return []
    predictions = []
    for peer, peer_df in aligned.items():
        if peer == station or timestamp not in peer_df.index:
            continue
        row = peer_df.loc[timestamp]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not bool(row["_peer_usable"]) or not np.isfinite(row["_kt"]):
            continue
        calibration = calibrations.get((station, peer, timestamp.month))
        if calibration is None:
            calibration = calibrations.get((station, peer, "all"))
        if calibration is None:
            continue
        predicted_kt = np.clip(
            calibration["intercept"] + calibration["slope"] * float(row["_kt"]),
            0,
            config.max_kt,
        )
        prediction = float(predicted_kt * target_rso)
        uncertainty = float(max(calibration["scatter_kt"] * target_rso, 10.0))
        predictions.append(
            {
                "peer": peer,
                "value": prediction,
                "uncertainty": uncertainty,
                "weight": 1.0 / max(uncertainty**2, 1.0),
            }
        )
    return predictions


def combine_peers(predictions: list[dict], config: Config) -> dict | None:
    """Reject inconsistent peer estimates and combine the remainder robustly."""
    if not predictions:
        return None
    values = np.array([item["value"] for item in predictions], dtype=float)
    weights = np.array([item["weight"] for item in predictions], dtype=float)
    center = weighted_median(values, weights)
    spread = robust_scale(values)
    if not np.isfinite(spread):
        spread = 0.0
    consistent_mask = np.abs(values - center) <= max(
        config.peer_consistency_abs_wm2,
        config.peer_consistency_relative * max(center, 50.0),
    )
    kept = [item for item, keep in zip(predictions, consistent_mask) if keep]
    if not kept:
        kept = [predictions[int(np.argmin(np.abs(values - center)))]]
    kept_values = np.array([item["value"] for item in kept])
    kept_weights = np.array([item["weight"] for item in kept])
    estimate = weighted_median(kept_values, kept_weights)
    kept_spread = robust_scale(kept_values) if len(kept) > 1 else kept[0]["uncertainty"]
    calibration_uncertainty = weighted_median(
        np.array([item["uncertainty"] for item in kept]), kept_weights
    )
    uncertainty = float(math.sqrt(max(kept_spread, 0) ** 2 + calibration_uncertainty**2))
    return {
        "value": estimate,
        "uncertainty": uncertainty,
        "spread": float(kept_spread),
        "sources": [item["peer"] for item in kept],
        "peer_count": len(kept),
        "all_peer_count": len(predictions),
    }


def temporal_kt_estimate(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    target_rso: float,
    invalid_times: set[pd.Timestamp],
    config: Config,
) -> dict | None:
    """Interpolate the target clear-sky index only across short, bounded gaps."""
    if not np.isfinite(target_rso) or target_rso < config.min_rso_wm2:
        return None
    valid = df["_normal"] & ~df.index.isin(invalid_times)
    previous = df.loc[(df.index < timestamp) & valid, "_kt"].tail(1)
    following = df.loc[(df.index > timestamp) & valid, "_kt"].head(1)
    if previous.empty or following.empty:
        return None
    before_time, after_time = previous.index[0], following.index[0]
    missing_hours = int((after_time - before_time).total_seconds() / 3600) - 1
    if missing_hours < 1 or missing_hours > config.max_temporal_gap_hours:
        return None
    fraction = (timestamp - before_time) / (after_time - before_time)
    kt = float(previous.iloc[0] + fraction * (following.iloc[0] - previous.iloc[0]))
    uncertainty = float(abs(following.iloc[0] - previous.iloc[0]) * target_rso / 2 + 15)
    return {
        "value": kt * target_rso,
        "uncertainty": uncertainty,
        "sources": [f"{before_time}", f"{after_time}"],
        "peer_count": 0,
    }


def climatology_estimate(
    station: str,
    timestamp: pd.Timestamp,
    target_rso: float,
    climatology: dict[str, pd.DataFrame],
    config: Config,
) -> dict | None:
    """Estimate Srad from the target station's month-hour clear-sky climatology."""
    if not np.isfinite(target_rso) or target_rso < config.min_rso_wm2:
        return None
    key = (timestamp.month, timestamp.hour)
    table = climatology[station]
    if key not in table.index:
        return None
    row = table.loc[key]
    if int(row["count"]) < config.min_climatology_records or not np.isfinite(row["median"]):
        return None
    scatter = float(row["scatter"]) if np.isfinite(row["scatter"]) else 0.5
    return {
        "value": float(row["median"] * target_rso),
        "uncertainty": float(max(scatter * target_rso, 25)),
        "sources": [f"{station}:{timestamp.month:02d}-{timestamp.hour:02d} climatology"],
        "peer_count": 0,
    }


def empty_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize filled values and audit fields without changing observed Srad."""
    out = df.copy()
    out["Srad_original"] = out[SOLAR_COL]
    out["Srad_filled"] = out[SOLAR_COL]
    out["Srad_was_imputed"] = False
    out["Srad_was_corrected"] = False
    out["Srad_imputation_reason"] = "observed_unchanged"
    out["Srad_imputation_method"] = "none"
    out["Srad_imputation_sources"] = ""
    out["Srad_peer_count"] = 0
    out["Srad_imputation_uncertainty"] = np.nan
    out["Srad_review_status"] = "not_required"
    out["Srad_value_was_clipped"] = False
    return out


def set_result(
    out: pd.DataFrame,
    index: int,
    estimate: dict,
    reason: str,
    method: str,
    imputed: bool,
    corrected: bool,
    review_status: str,
    config: Config,
) -> None:
    """Store one imputation decision, enforce physical bounds, and update audit fields."""
    raw_value = float(estimate["value"])
    clipped = float(np.clip(raw_value, config.physical_min, config.physical_max))
    out.at[index, "Srad_filled"] = clipped
    out.at[index, "Srad_was_imputed"] = imputed
    out.at[index, "Srad_was_corrected"] = corrected
    out.at[index, "Srad_imputation_reason"] = reason
    out.at[index, "Srad_imputation_method"] = method
    out.at[index, "Srad_imputation_sources"] = ";".join(estimate.get("sources", []))
    out.at[index, "Srad_peer_count"] = int(estimate.get("peer_count", 0))
    out.at[index, "Srad_imputation_uncertainty"] = estimate.get("uncertainty", np.nan)
    out.at[index, "Srad_review_status"] = review_status
    out.at[index, "Srad_value_was_clipped"] = not np.isclose(raw_value, clipped)


def impute_all(
    datasets: dict[str, pd.DataFrame],
    config: Config,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Run the complete station imputation workflow and assemble logs and calibrations."""
    aligned = {station: add_model_columns(df, config).set_index(DATE_COL) for station, df in datasets.items()}
    calibrations = build_calibrations(aligned, config)
    climatology = build_climatology(aligned, config)
    calibration_rows = [
        {
            "target_station": target,
            "peer_station": peer,
            "month": month,
            **values,
        }
        for (target, peer, month), values in calibrations.items()
    ]
    results = {}
    log_rows = []

    for station, source in datasets.items():
        out = empty_audit(source)
        model_df = aligned[station]
        invalid_mask = (
            out[SOLAR_COL].isna()
            | ~out[SOLAR_COL].between(config.physical_min, config.physical_max)
            | out["missing_srad_flag"]
            | out["out_of_range_flag"]
        )
        invalid_times = set(out.loc[invalid_mask, DATE_COL])

        for idx, row in out.iterrows():
            timestamp = row[DATE_COL]
            original = row[SOLAR_COL]
            is_daytime = bool(row["_is_daytime"])
            is_missing = pd.isna(original) or bool(row["missing_srad_flag"])
            is_out_of_range = (
                pd.notna(original)
                and not config.physical_min <= float(original) <= config.physical_max
            ) or bool(row["out_of_range_flag"])
            peer_review = any(bool(row[column]) for column in PEER_REVIEW_FLAGS)
            night_anomaly = bool(row["night_radiation_anomaly_flag"])
            weather_event = bool(row["weather_related_low_radiation_event_flag"])

            if weather_event and not (is_missing or is_out_of_range or night_anomaly):
                out.at[idx, "Srad_imputation_reason"] = "weather_explained_low_radiation_retained"
                out.at[idx, "Srad_review_status"] = "retained_weather_explained"
                continue

            if night_anomaly and not is_daytime:
                set_result(
                    out,
                    idx,
                    {"value": 0.0, "uncertainty": 0.0, "sources": ["astronomical_night"], "peer_count": 0},
                    "astronomically_confirmed_night_anomaly",
                    "astronomical_night_zero",
                    imputed=is_missing,
                    corrected=not is_missing,
                    review_status="resolved_automatic",
                    config=config,
                )
                continue

            if (is_missing or is_out_of_range) and not is_daytime:
                set_result(
                    out,
                    idx,
                    {"value": 0.0, "uncertainty": 0.0, "sources": ["astronomical_night"], "peer_count": 0},
                    "missing_or_invalid_during_astronomical_night",
                    "astronomical_night_zero",
                    imputed=is_missing,
                    corrected=is_out_of_range,
                    review_status="resolved_automatic",
                    config=config,
                )
                continue

            target_rso = row["Rso_wm2"]
            combined = combine_peers(
                peer_predictions(station, timestamp, target_rso, aligned, calibrations, config),
                config,
            )

            if peer_review and not (is_missing or is_out_of_range):
                flags = [column.removesuffix("_flag") for column in PEER_REVIEW_FLAGS if bool(row[column])]
                reason = "+".join(flags)
                if combined and combined["peer_count"] >= 2:
                    deviation = abs(float(original) - combined["value"])
                    threshold = max(
                        config.obvious_deviation_abs_wm2,
                        config.obvious_deviation_relative * max(combined["value"], 50),
                        2 * combined["spread"],
                    )
                    if deviation >= threshold:
                        set_result(
                            out,
                            idx,
                            combined,
                            f"{reason}_peer_consensus_and_clear_deviation",
                            "multi_peer_clear_sky_index",
                            imputed=False,
                            corrected=True,
                            review_status="resolved_peer_consensus",
                            config=config,
                        )
                    else:
                        out.at[idx, "Srad_imputation_reason"] = f"{reason}_retained_not_clearly_deviant"
                        out.at[idx, "Srad_imputation_method"] = "peer_consensus_check_only"
                        out.at[idx, "Srad_imputation_sources"] = ";".join(combined["sources"])
                        out.at[idx, "Srad_peer_count"] = combined["peer_count"]
                        out.at[idx, "Srad_imputation_uncertainty"] = combined["uncertainty"]
                        out.at[idx, "Srad_review_status"] = "manual_review_retained"
                else:
                    out.at[idx, "Srad_imputation_reason"] = f"{reason}_retained_insufficient_peer_consensus"
                    out.at[idx, "Srad_imputation_method"] = "none"
                    if combined:
                        out.at[idx, "Srad_imputation_sources"] = ";".join(combined["sources"])
                        out.at[idx, "Srad_peer_count"] = combined["peer_count"]
                        out.at[idx, "Srad_imputation_uncertainty"] = combined["uncertainty"]
                    out.at[idx, "Srad_review_status"] = "manual_review_retained"
                continue

            if not (is_missing or is_out_of_range):
                continue

            reason = "missing_Srad" if is_missing else "out_of_range_Srad"
            estimate = None
            method = "none"
            if combined and combined["peer_count"] >= 2:
                estimate, method = combined, "multi_peer_clear_sky_index"
            elif combined and combined["peer_count"] == 1:
                estimate, method = combined, "single_peer_clear_sky_index"
            if estimate is None:
                estimate = temporal_kt_estimate(
                    model_df, timestamp, target_rso, invalid_times, config
                )
                method = "temporal_clear_sky_index_interpolation" if estimate else "none"
            if estimate is None:
                estimate = climatology_estimate(
                    station, timestamp, target_rso, climatology, config
                )
                method = "month_hour_climatology" if estimate else "none"
            if estimate is not None:
                set_result(
                    out,
                    idx,
                    estimate,
                    reason,
                    method,
                    imputed=is_missing,
                    corrected=is_out_of_range,
                    review_status="resolved_automatic",
                    config=config,
                )
            else:
                out.at[idx, "Srad_filled"] = np.nan
                out.at[idx, "Srad_imputation_reason"] = f"{reason}_no_reliable_estimate"
                out.at[idx, "Srad_imputation_method"] = "unfilled"
                out.at[idx, "Srad_review_status"] = "manual_review_unfilled"

        public = out.drop(columns=["_sunrise", "_sunset", "_is_daytime"], errors="ignore")
        results[station] = public
        log_mask = (
            public["Srad_was_imputed"]
            | public["Srad_was_corrected"]
            | public["Srad_review_status"].str.startswith("manual_review")
        )
        for _, row in public.loc[log_mask].iterrows():
            log_rows.append(
                {
                    "station": station,
                    "Date": row[DATE_COL],
                    "Srad_original": row["Srad_original"],
                    "Srad_filled": row["Srad_filled"],
                    "was_imputed": row["Srad_was_imputed"],
                    "was_corrected": row["Srad_was_corrected"],
                    "reason": row["Srad_imputation_reason"],
                    "method": row["Srad_imputation_method"],
                    "sources": row["Srad_imputation_sources"],
                    "peer_count": row["Srad_peer_count"],
                    "uncertainty": row["Srad_imputation_uncertainty"],
                    "review_status": row["Srad_review_status"],
                    "value_was_clipped": row["Srad_value_was_clipped"],
                }
            )
    return results, pd.DataFrame(log_rows), pd.DataFrame(calibration_rows)


def choose_validation_segments(
    aligned: dict[str, pd.DataFrame], config: Config
) -> tuple[pd.DataFrame, dict[str, set[pd.Timestamp]]]:
    """Select reproducible, nonoverlapping normal-data gaps for validation masking."""
    rng = np.random.default_rng(RANDOM_SEED)
    buckets = {
        "1 hour": (1, 1),
        "2-6 hours": (2, 6),
        "7-24 hours": (7, 24),
        "multi-day": (48, 96),
    }
    rows = []
    excluded: dict[str, set[pd.Timestamp]] = {station: set() for station in aligned}
    for station, df in aligned.items():
        index = df.index
        normal = df["_validation_normal"].to_numpy()
        for bucket, (low, high) in buckets.items():
            candidates = np.arange(1, max(len(df) - high - 1, 1))
            rng.shuffle(candidates)
            selected = 0
            for start_pos in candidates:
                length = int(rng.integers(low, high + 1))
                end_pos = start_pos + length
                if end_pos >= len(df) - 1:
                    continue
                segment_times = index[start_pos:end_pos]
                if len(segment_times) != length:
                    continue
                expected = pd.date_range(segment_times[0], periods=length, freq="h")
                if not segment_times.equals(expected):
                    continue
                if not normal[start_pos:end_pos].all():
                    continue
                if any(time in excluded[station] for time in index[start_pos - 1 : end_pos + 1]):
                    continue
                segment_id = f"{station}_{bucket.replace(' ', '_')}_{selected + 1:02d}"
                for time in segment_times:
                    rows.append(
                        {
                            "segment_id": segment_id,
                            "station": station,
                            "gap_bucket": bucket,
                            "gap_hours": length,
                            "Date": time,
                            "actual": float(df.at[time, SOLAR_COL]),
                        }
                    )
                    excluded[station].add(time)
                selected += 1
                if selected >= config.validation_segments_per_bucket:
                    break
    return pd.DataFrame(rows), excluded


def validation_predictions(
    validation_rows: pd.DataFrame,
    aligned: dict[str, pd.DataFrame],
    excluded: dict[str, set[pd.Timestamp]],
    calibrations: dict,
    climatology: dict,
    config: Config,
) -> pd.DataFrame:
    """Apply every comparison method to masked validation records and collect predictions."""
    output = []
    for (station, segment_id), segment in validation_rows.groupby(["station", "segment_id"]):
        df = aligned[station]
        times = pd.DatetimeIndex(segment[DATE_COL])
        first, last = times.min(), times.max()
        before = df.loc[(df.index < first) & df["_validation_normal"], SOLAR_COL].tail(1)
        after = df.loc[(df.index > last) & df["_validation_normal"], SOLAR_COL].head(1)
        linear_values: dict[pd.Timestamp, float] = {}
        if not before.empty and not after.empty:
            x0, x1 = before.index[0].value, after.index[0].value
            for timestamp in times:
                fraction = (timestamp.value - x0) / (x1 - x0)
                linear_values[timestamp] = float(before.iloc[0] + fraction * (after.iloc[0] - before.iloc[0]))

        before_kt = df.loc[(df.index < first) & df["_normal"], "_kt"].tail(1)
        after_kt = df.loc[(df.index > last) & df["_normal"], "_kt"].head(1)
        kt_values: dict[pd.Timestamp, float] = {}
        if not before_kt.empty and not after_kt.empty:
            x0, x1 = before_kt.index[0].value, after_kt.index[0].value
            for timestamp in times:
                rso = df.at[timestamp, "Rso_wm2"]
                if np.isfinite(rso) and rso >= config.min_rso_wm2:
                    fraction = (timestamp.value - x0) / (x1 - x0)
                    kt = float(before_kt.iloc[0] + fraction * (after_kt.iloc[0] - before_kt.iloc[0]))
                    kt_values[timestamp] = kt * rso
                elif not bool(df.at[timestamp, "_is_daytime"]):
                    kt_values[timestamp] = 0.0

        for _, record in segment.iterrows():
            timestamp = record[DATE_COL]
            rso = df.at[timestamp, "Rso_wm2"]
            methods: dict[str, float | None] = {
                "linear_Srad_interpolation": linear_values.get(timestamp),
                "clear_sky_index_interpolation": kt_values.get(timestamp),
            }
            if not bool(df.at[timestamp, "_is_daytime"]):
                methods.update(
                    {
                        "month_hour_climatology": 0.0,
                        "single_peer": 0.0,
                        "multi_peer": 0.0,
                    }
                )
            else:
                climate = climatology_estimate(station, timestamp, rso, climatology, config)
                predictions = peer_predictions(station, timestamp, rso, aligned, calibrations, config)
                combined = combine_peers(predictions, config)
                single = min(predictions, key=lambda item: item["uncertainty"]) if predictions else None
                methods.update(
                    {
                        "month_hour_climatology": climate["value"] if climate else None,
                        "single_peer": single["value"] if single else None,
                        "multi_peer": combined["value"] if combined and combined["peer_count"] >= 2 else None,
                    }
                )
            for method, prediction in methods.items():
                output.append(
                    {
                        **record.to_dict(),
                        "method": method,
                        "prediction": np.clip(prediction, 0, config.physical_max)
                        if prediction is not None and np.isfinite(prediction)
                        else np.nan,
                    }
                )
    return pd.DataFrame(output)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate coverage, MAE, RMSE, bias, and R-squared at each reporting level."""
    rows = []
    groupings = [
        (["station", "gap_bucket", "method"], "station_gap"),
        (["gap_bucket", "method"], "all_stations_gap"),
        (["method"], "overall"),
    ]
    for columns, scope in groupings:
        grouper = columns[0] if len(columns) == 1 else columns
        for keys, group in predictions.groupby(grouper, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            valid = group.dropna(subset=["actual", "prediction"])
            n_total = len(group)
            n = len(valid)
            row = {"scope": scope, "station": "ALL", "gap_bucket": "ALL", "method": ""}
            row.update(dict(zip(columns, keys)))
            row["n_expected"] = n_total
            row["n_predicted"] = n
            row["coverage"] = n / n_total if n_total else np.nan
            if n:
                error = valid["prediction"].to_numpy() - valid["actual"].to_numpy()
                actual = valid["actual"].to_numpy()
                predicted = valid["prediction"].to_numpy()
                row["MAE"] = float(np.mean(np.abs(error)))
                row["RMSE"] = float(np.sqrt(np.mean(error**2)))
                row["bias"] = float(np.mean(error))
                denominator = float(np.sum((actual - actual.mean()) ** 2))
                row["R2"] = 1 - float(np.sum(error**2)) / denominator if denominator > 0 else np.nan
            else:
                row.update({"MAE": np.nan, "RMSE": np.nan, "bias": np.nan, "R2": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def build_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize imputation, correction, review, and clipping counts by station and method."""
    rows = []
    for station, df in results.items():
        for method, group in df.groupby("Srad_imputation_method", dropna=False):
            rows.append(
                {
                    "station": station,
                    "method": method,
                    "records": len(group),
                    "imputed": int(group["Srad_was_imputed"].sum()),
                    "corrected": int(group["Srad_was_corrected"].sum()),
                    "unfilled": int((group["Srad_imputation_method"] == "unfilled").sum()),
                    "manual_review_retained": int(
                        (group["Srad_review_status"] == "manual_review_retained").sum()
                    ),
                    "clipped": int(group["Srad_value_was_clipped"].sum()),
                }
            )
    return pd.DataFrame(rows)


def verify_outputs(
    datasets: dict[str, pd.DataFrame], results: dict[str, pd.DataFrame], config: Config
) -> pd.DataFrame:
    """Verify row and timestamp integrity, original-value preservation, and physical bounds."""
    rows = []
    for station in sorted(datasets):
        source, result = datasets[station], results[station]
        any_flag = pd.Series(False, index=source.index)
        for column in FLAG_COLUMNS:
            any_flag |= source[column]
        normal_unmarked = (
            ~any_flag
            & source[SOLAR_COL].notna()
            & source[SOLAR_COL].between(config.physical_min, config.physical_max)
        )
        unchanged_normal = np.allclose(
            source.loc[normal_unmarked, SOLAR_COL],
            result.loc[normal_unmarked, "Srad_filled"],
            equal_nan=True,
        )
        original_preserved = np.allclose(
            source[SOLAR_COL], result["Srad_original"], equal_nan=True
        )
        rows.append(
            {
                "station": station,
                "input_rows": len(source),
                "output_rows": len(result),
                "row_count_match": len(source) == len(result),
                "timestamp_match": source[DATE_COL].equals(result[DATE_COL]),
                "Srad_original_preserved": original_preserved,
                "normal_unmarked_records": int(normal_unmarked.sum()),
                "normal_unmarked_unchanged": unchanged_normal,
                "filled_outside_physical_range": int(
                    (
                        result["Srad_filled"].notna()
                        & ~result["Srad_filled"].between(config.physical_min, config.physical_max)
                    ).sum()
                ),
            }
        )
    checks = pd.DataFrame(rows)
    boolean_columns = [
        "row_count_match",
        "timestamp_match",
        "Srad_original_preserved",
        "normal_unmarked_unchanged",
    ]
    if not checks[boolean_columns].all().all() or checks["filled_outside_physical_range"].sum():
        raise AssertionError(f"Output verification failed:\n{checks.to_string(index=False)}")
    return checks


def make_figures(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Create figures summarizing station outcomes and validation RMSE."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    totals = summary.groupby("station")[["imputed", "corrected", "unfilled", "manual_review_retained"]].sum()
    ax = totals.plot.bar(figsize=(10, 5), color=["#0b7fb8", "#08a678", "#e66a00", "#7a8a99"])
    ax.set_title("Solar-radiation imputation and review outcomes")
    ax.set_ylabel("Records")
    ax.set_xlabel("Station")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "imputation_outcomes_by_station.png", dpi=180)
    plt.close()

    overall = metrics[(metrics["scope"] == "all_stations_gap")].copy()
    pivot = overall.pivot(index="gap_bucket", columns="method", values="RMSE")
    order = [item for item in ["1 hour", "2-6 hours", "7-24 hours", "multi-day"] if item in pivot.index]
    pivot = pivot.reindex(order)
    ax = pivot.plot.bar(figsize=(12, 6))
    ax.set_title("Validation RMSE by masked-gap length")
    ax.set_ylabel("RMSE (W/m²)")
    ax.set_xlabel("Masked gap")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.legend(title="Method", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "validation_rmse_by_method.png", dpi=180)
    plt.close()


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Format selected DataFrame columns as a compact Markdown table."""
    display = df.loc[:, columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}" if isinstance(value, float) else str(value)
        )
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, rule, *lines])


def write_report(
    input_checks: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    verification: pd.DataFrame,
    results: dict[str, pd.DataFrame],
) -> None:
    """Write the final Markdown report from input checks, outcomes, and validation results."""
    station_totals = summary.groupby("station")[
        ["imputed", "corrected", "unfilled", "manual_review_retained", "clipped"]
    ].sum().reset_index()
    overall = metrics[metrics["scope"] == "overall"].sort_values("RMSE")
    review_rows = []
    for station, df in results.items():
        retained = df[df["Srad_review_status"] == "manual_review_retained"]
        for reason, group in retained.groupby("Srad_imputation_reason"):
            review_rows.append({"station": station, "reason": reason, "records": len(group)})
    review = pd.DataFrame(review_rows, columns=["station", "reason", "records"])
    best = overall.iloc[0] if not overall.empty else None
    best_method = best["method"] if best is not None else "not available"
    best_rmse = f'{best["RMSE"]:.3f} W/m²' if best is not None else "not available"
    report = f"""# Solar Radiation Imputation Report

Generated with fixed random seed `{RANDOM_SEED}`.

## Inputs and units

Six hourly anomaly-marked station files were checked against their corresponding
prewashed source files. `Rso` was detected from its observed range as
`MJ m-2 h-1` and converted using `Rso_wm2 = Rso × {RSO_MJ_TO_WM2}`.  Astronomical
day/night status used station coordinates, the `{TIMEZONE}` timezone, and a
30-minute sunrise/sunset buffer.

{markdown_table(input_checks, ["station", "rows", "missing_Srad", "Rso_min", "Rso_max", "Rso_detected_unit", "raw_timestamp_match", "raw_Srad_match"])}

## Imputation outcomes

Observed `Srad` is retained in both `Srad` and `Srad_original`; only
`Srad_filled` contains a replacement. Weather-explained low radiation is
retained. Long-zero, sudden-drop, and spike records are replaced only when at
least two calibrated peer predictions agree and the observation is clearly
deviant; all others remain unchanged and are marked for manual review.

{markdown_table(station_totals, ["station", "imputed", "corrected", "unfilled", "manual_review_retained", "clipped"])}

![Imputation outcomes](figures/imputation_outcomes_by_station.png)

## Validation

Normal records were masked in reproducible continuous segments of 1 hour, 2–6
hours, 7–24 hours, and 48–96 hours. Calibration and climatology fitting excluded
all target-station validation timestamps. The table below aggregates all gap
lengths and stations.

{markdown_table(overall, ["method", "n_expected", "n_predicted", "coverage", "MAE", "RMSE", "bias", "R2"])}

Best overall RMSE: **{best_method}** ({best_rmse}).

![Validation RMSE](figures/validation_rmse_by_method.png)

Detailed station-by-gap metrics are in `validation_metrics.csv`, and all masked
predictions are in `validation_predictions.csv`.

## Manual review retained

{markdown_table(review, ["station", "reason", "records"]) if not review.empty else "No peer-dependent anomalies were retained for manual review."}

## Integrity checks

{markdown_table(verification, ["station", "input_rows", "output_rows", "row_count_match", "timestamp_match", "Srad_original_preserved", "normal_unmarked_records", "normal_unmarked_unchanged", "filled_outside_physical_range"])}

## Method notes

- Monthly robust affine relationships map peer-station clear-sky index to the
  target station; global relationships are used only when a month lacks enough
  paired normal records.
- Priority is multi-peer, single-peer, clear-sky-index interpolation for gaps
  no longer than three hours, then month-hour climatology.
- No raw-`Srad` interpolation is used in production imputation, and unreliable
  values remain `NA`.
- Generated values are constrained to 0–1300 W/m²; clipping is recorded in
  `Srad_value_was_clipped`.
"""
    (OUTPUT_DIR / "Solar_Radiation_Imputation_Report.md").write_text(report, encoding="utf-8")


def run(config: Config) -> None:
    """Execute the full workflow and write all data, validation, figure, and report outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FILLED_DIR.mkdir(parents=True, exist_ok=True)
    datasets, input_checks = load_inputs(config)
    results, log, calibrations = impute_all(datasets, config)
    summary = build_summary(results)
    verification = verify_outputs(datasets, results, config)

    aligned = {station: add_model_columns(df, config).set_index(DATE_COL) for station, df in datasets.items()}
    validation_rows, excluded = choose_validation_segments(aligned, config)
    validation_calibrations = build_calibrations(aligned, config, excluded)
    validation_climatology = build_climatology(aligned, config, excluded)
    validation_predictions_df = validation_predictions(
        validation_rows,
        aligned,
        excluded,
        validation_calibrations,
        validation_climatology,
        config,
    )
    metrics = metric_rows(validation_predictions_df)

    for station, result in results.items():
        result.to_csv(FILLED_DIR / f"{station}_met_srad_imputed.csv", index=False)
    input_checks.to_csv(OUTPUT_DIR / "input_data_checks.csv", index=False)
    log.to_csv(OUTPUT_DIR / "imputation_log.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "imputation_summary_by_station_method.csv", index=False)
    calibrations.to_csv(OUTPUT_DIR / "peer_calibration_parameters.csv", index=False)
    validation_rows.to_csv(OUTPUT_DIR / "validation_masked_segments.csv", index=False)
    validation_predictions_df.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    verification.to_csv(OUTPUT_DIR / "output_integrity_checks.csv", index=False)
    make_figures(summary, metrics)
    write_report(input_checks, summary, metrics, verification, results)

    manifest = {
        "random_seed": RANDOM_SEED,
        "stations": sorted(results),
        "output_audit_columns": OUTPUT_AUDIT_COLUMNS,
        "rso_conversion_factor": RSO_MJ_TO_WM2,
        "physical_bounds_wm2": [config.physical_min, config.physical_max],
        "validation_segments_per_station_bucket_requested": config.validation_segments_per_bucket,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options controlling the reproducible validation sample size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-segments-per-bucket",
        type=int,
        default=Config.validation_segments_per_bucket,
        help="Number of non-overlapping validation gaps sampled per station and gap bucket.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Config(validation_segments_per_bucket=args.validation_segments_per_bucket))

"""Build the modeling-only complete TxSON33 feature matrix.

This is intentionally separate from the original imputation pipeline. It
preserves every finite authoritative value and fills only structural or
out-of-coverage missing cells for workflows that require a complete matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "data-cleanup" / "imputation_pipeline"
AUTHORITATIVE_DIR = ROOT / "modeling_data" / "txson33_authoritative_2026-09-07"
AUTHORITATIVE_FILE = AUTHORITATIVE_DIR / "TxSON33_modeling.parquet"
STAGE0_DIR = PIPELINE_DIR / "cleaned_data"
OUTPUT_DIR = ROOT / "datasets" / "txson33_complete_for_modeling"
REVIEW_DIR = ROOT / "review_only" / "complete_modeling_dataset_2026-09-12"

SOIL_MOISTURE = ["SWC_5", "SWC_10", "SWC_20", "SWC_50"]
SOIL_TEMPERATURE = ["T_5", "T_10", "T_20", "T_50"]
SOIL_VARIABLES = SOIL_MOISTURE + SOIL_TEMPERATURE
MET_COMPLETION_VARIABLES = ["Tair", "RH", "Srad", "Wind speed", "Wind direction"]
MEASUREMENT_VARIABLES = SOIL_VARIABLES + MET_COMPLETION_VARIABLES + ["Ppt"]
DEDICATED_MET_STATIONS = {"CB01", "CB04", "CB06", "FD02", "FD03", "WC05"}
EXPECTED_ABSENT_SOIL = {
    (station, variable)
    for station in ["CB07", "CB26", "FD03", "FD18", "FD21", "FD24"]
    for variable in ["SWC_50", "T_50"]
}
PENDING_QC: set[tuple[str, str]] = set()
MANUAL_QC_FILE = PIPELINE_DIR / "manual_qc_masks.csv"
SENSOR_QC_DECISIONS_FILE = PIPELINE_DIR / "sensor_qc_review_decisions.csv"
AUTHORITATIVE_PROVENANCE_FILE = AUTHORITATIVE_DIR / "imputation_provenance.csv"

OBSERVED = 0
PIPELINE_IMPUTED = 1
MODELING_COMPLETION = 2
ORIGIN_LABELS = ["observed", "pipeline_imputed", "modeling_completion"]
RANDOM_SEED = 20260912
MAX_TRAIN_ROWS = 300_000
MAX_VALIDATION_ROWS = 50_000
PREDICTION_CHUNK_ROWS = 250_000
EXPECTED_ROWS = 2_720_775

NEW_VALUE_RANGES = {
    "SWC_5": (0.0, 0.6),
    "SWC_10": (0.0, 0.6),
    "SWC_20": (0.0, 0.6),
    "SWC_50": (0.0, 0.6),
    "T_5": (-30.0, 60.0),
    "T_10": (-30.0, 60.0),
    "T_20": (-30.0, 60.0),
    "T_50": (-30.0, 60.0),
    "Tair": (-40.0, 60.0),
    "RH": (0.0, 100.0),
    "Srad": (0.0, 1500.0),
    "Wind speed": (0.0, 75.0),
    "Ppt": (0.0, 500.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_snapshot(paths: list[Path]) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for root in paths:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def deterministic_sample(indices: np.ndarray, limit: int, seed_offset: int) -> np.ndarray:
    if len(indices) <= limit:
        return np.sort(indices)
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def clip_new_values(variable: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if variable == "Wind direction":
        return np.mod(values, 360.0)
    lower, upper = NEW_VALUE_RANGES[variable]
    return np.clip(values, lower, upper)


def circular_mean(values: np.ndarray, axis: int = 1) -> np.ndarray:
    radians = np.deg2rad(values)
    valid_count = np.isfinite(radians).sum(axis=axis)
    sin_mean = np.divide(
        np.nansum(np.sin(radians), axis=axis),
        valid_count,
        out=np.full_like(valid_count, np.nan, dtype=float),
        where=valid_count > 0,
    )
    cos_mean = np.divide(
        np.nansum(np.cos(radians), axis=axis),
        valid_count,
        out=np.full_like(valid_count, np.nan, dtype=float),
        where=valid_count > 0,
    )
    resultant = np.hypot(sin_mean, cos_mean)
    result = np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0)
    result[~np.isfinite(result) | (resultant < 1e-8)] = np.nan
    return result


def row_nanmedian(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    result = np.full(len(values), np.nan, dtype=float)
    rows = valid.any(axis=1)
    if rows.any():
        result[rows] = np.nanmedian(values[rows], axis=1)
    return result


def circular_errors(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.abs((prediction - truth + 180.0) % 360.0 - 180.0)


def error_metrics(truth: np.ndarray, prediction: np.ndarray, circular: bool = False) -> dict:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return {"N": 0, "MAE": np.nan, "RMSE": np.nan}
    if circular:
        errors = circular_errors(truth[valid], prediction[valid])
    else:
        errors = np.abs(prediction[valid] - truth[valid])
    return {
        "N": int(valid.sum()),
        "MAE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
    }


def apply_qc_origin_overrides(frame: pd.DataFrame, origins: dict[str, np.ndarray]) -> None:
    manual = pd.read_csv(MANUAL_QC_FILE, parse_dates=["Start", "End"])
    manual = manual.loc[manual["Decision"] == "mask_and_refill"]
    for rule in manual.itertuples(index=False):
        if rule.Parameter not in origins:
            raise ValueError(f"Unknown manual-QC parameter: {rule.Parameter}")
        affected = (
            (frame["Station"] == rule.Station)
            & (frame["Timestamp"] >= rule.Start)
            & (frame["Timestamp"] <= rule.End)
            & frame[rule.Parameter].notna()
        )
        origins[rule.Parameter][affected.to_numpy()] = PIPELINE_IMPUTED

    decisions = pd.read_csv(SENSOR_QC_DECISIONS_FILE)
    approved = decisions.loc[decisions["Decision"] == "approved"]
    for rule in approved.itertuples(index=False):
        if rule.Parameter not in origins:
            raise ValueError(f"Unknown sensor-QC parameter: {rule.Parameter}")
        affected = (frame["Station"] == rule.Station) & frame[rule.Parameter].notna()
        origins[rule.Parameter][affected.to_numpy()] = PIPELINE_IMPUTED


def validate_authoritative_provenance(stage0_filled_counts: dict[tuple[str, str], int]) -> None:
    provenance_counts: dict[tuple[str, str], int] = {}
    for chunk in pd.read_csv(
        AUTHORITATIVE_PROVENANCE_FILE,
        usecols=["Station", "Parameter"],
        chunksize=500_000,
    ):
        for key, count in chunk.value_counts(["Station", "Parameter"]).items():
            provenance_counts[(str(key[0]), str(key[1]))] = (
                provenance_counts.get((str(key[0]), str(key[1])), 0) + int(count)
            )
    expected = {key: count for key, count in stage0_filled_counts.items() if count}
    if provenance_counts != expected:
        missing = sorted(set(expected) - set(provenance_counts))[:10]
        extra = sorted(set(provenance_counts) - set(expected))[:10]
        mismatched = sorted(
            key for key in set(expected) & set(provenance_counts)
            if expected[key] != provenance_counts[key]
        )[:10]
        raise ValueError(
            "Authoritative imputation provenance does not match Stage 0/final lineage: "
            f"missing={missing}, extra={extra}, count_mismatches={mismatched}"
        )


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, list[dict], set[tuple[str, str]]]:
    frame = pd.read_parquet(AUTHORITATIVE_FILE)
    missing_columns = [column for column in ["Station", "Timestamp", *MEASUREMENT_VARIABLES] if column not in frame]
    if missing_columns:
        raise ValueError(f"Authoritative dataset is missing columns: {missing_columns}")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="raise")
    if frame.duplicated(["Station", "Timestamp"]).any():
        raise ValueError("Authoritative dataset contains duplicate Station/Timestamp rows")

    n_rows = len(frame)
    origins = {variable: np.full(n_rows, MODELING_COMPLETION, dtype=np.uint8) for variable in MEASUREMENT_VARIABLES}
    missing_structure: list[dict] = []
    source_absent: set[tuple[str, str]] = set()
    dedicated_stage0_parts = []
    stage0_filled_counts: dict[tuple[str, str], int] = {}

    for station, positions in frame.groupby("Station", sort=False).indices.items():
        positions = np.asarray(positions)
        station_rows = frame.iloc[positions]
        station_index = pd.DatetimeIndex(station_rows["Timestamp"])
        baseline_path = STAGE0_DIR / f"Station{station}_cleaned_data.csv"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing authoritative Stage 0 input: {baseline_path}")
        baseline = pd.read_csv(baseline_path, parse_dates=["Date"]).set_index("Date")
        if baseline.index.has_duplicates:
            raise ValueError(f"{baseline_path.name} has duplicate timestamps")
        baseline = baseline.reindex(station_index)

        if station in DEDICATED_MET_STATIONS:
            met_observed = baseline.reindex(columns=MET_COMPLETION_VARIABLES).copy()
            met_observed.insert(0, "Timestamp", station_index)
            met_observed.insert(0, "Station", station)
            dedicated_stage0_parts.append(met_observed.reset_index(drop=True))

        for variable in MEASUREMENT_VARIABLES:
            authoritative = station_rows[variable].to_numpy(dtype=float)
            original = (
                baseline[variable].to_numpy(dtype=float)
                if variable in baseline
                else np.full(len(baseline), np.nan)
            )
            authoritative_finite = np.isfinite(authoritative)
            original_finite = np.isfinite(original)
            unchanged = authoritative_finite & original_finite & (authoritative == original)
            pipeline_imputed = authoritative_finite & ~unchanged
            stage0_filled_counts[(station, variable)] = int(
                (authoritative_finite & ~original_finite).sum()
            )
            origins[variable][positions[unchanged]] = OBSERVED
            origins[variable][positions[pipeline_imputed]] = PIPELINE_IMPUTED

            if not original_finite.any():
                source_absent.add((station, variable))

            missing = ~authoritative_finite
            if variable in SOIL_VARIABLES:
                if original_finite.any():
                    source_start = station_index[original_finite].min()
                    source_end = station_index[original_finite].max()
                    outside = missing & ((station_index < source_start) | (station_index > source_end))
                    internal = missing & ~outside
                    missing_structure.extend([
                        {
                            "Station": station,
                            "Variable": variable,
                            "Category": "soil_outside_source_coverage",
                            "Missing_Count": int(outside.sum()),
                        },
                        {
                            "Station": station,
                            "Variable": variable,
                            "Category": "soil_other_remaining",
                            "Missing_Count": int(internal.sum()),
                        },
                    ])
                else:
                    missing_structure.append({
                        "Station": station,
                        "Variable": variable,
                        "Category": "soil_source_sensor_absent",
                        "Missing_Count": int(missing.sum()),
                    })
            elif variable in MET_COMPLETION_VARIABLES:
                category = (
                    "met_structurally_unavailable"
                    if station not in DEDICATED_MET_STATIONS
                    else "met_dedicated_station_remaining"
                )
                missing_structure.append({
                    "Station": station,
                    "Variable": variable,
                    "Category": category,
                    "Missing_Count": int(missing.sum()),
                })
            elif variable == "Ppt":
                missing_structure.append({
                    "Station": station,
                    "Variable": variable,
                    "Category": "ppt_remaining",
                    "Missing_Count": int(missing.sum()),
                })

    dedicated_stage0 = pd.concat(dedicated_stage0_parts, ignore_index=True)
    validate_authoritative_provenance(stage0_filled_counts)
    apply_qc_origin_overrides(frame, origins)
    return frame, origins, dedicated_stage0, missing_structure, source_absent


def validate_known_structure(
    frame: pd.DataFrame,
    missing_structure: list[dict],
    source_absent: set[tuple[str, str]],
) -> pd.DataFrame:
    if set(frame["Station"].unique()) & DEDICATED_MET_STATIONS != DEDICATED_MET_STATIONS:
        raise ValueError("One or more expected dedicated-MET stations are absent")
    actual_absent_soil = {(station, variable) for station, variable in source_absent if variable in SOIL_VARIABLES}
    if actual_absent_soil != EXPECTED_ABSENT_SOIL:
        raise ValueError(
            "Unexpected absent Soil sensor set: "
            f"expected={sorted(EXPECTED_ABSENT_SOIL)}, actual={sorted(actual_absent_soil)}"
        )
    structure = pd.DataFrame(missing_structure)
    categorized = int(structure["Missing_Count"].sum())
    actual = int(frame[MEASUREMENT_VARIABLES].isna().sum().sum())
    if categorized != actual:
        raise ValueError(f"Missing-category count {categorized} does not match actual missing count {actual}")
    return structure


def doy_hour_keys(timestamps: pd.Series | pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    index = pd.DatetimeIndex(timestamps)
    return index.dayofyear.to_numpy(), index.hour.to_numpy()


def pooled_met_climatology(
    observed: pd.DataFrame,
    variable: str,
    excluded_station: str | None = None,
) -> tuple[pd.Series, float]:
    data = observed if excluded_station is None else observed.loc[observed["Station"] != excluded_station]
    data = data.loc[data[variable].notna(), ["Timestamp", variable]].copy()
    data["DOY"] = data["Timestamp"].dt.dayofyear
    data["Hour"] = data["Timestamp"].dt.hour
    if variable == "Wind direction":
        data["Sin"] = np.sin(np.deg2rad(data[variable]))
        data["Cos"] = np.cos(np.deg2rad(data[variable]))
        grouped = data.groupby(["DOY", "Hour"])[["Sin", "Cos"]].mean()
        climatology = np.mod(np.rad2deg(np.arctan2(grouped["Sin"], grouped["Cos"])), 360.0)
        global_value = float(
            np.mod(
                np.rad2deg(np.arctan2(data["Sin"].mean(), data["Cos"].mean())),
                360.0,
            )
        )
    else:
        climatology = data.groupby(["DOY", "Hour"])[variable].median()
        global_value = float(data[variable].median())
    return climatology, global_value


def lookup_climatology(
    timestamps: pd.Series | pd.DatetimeIndex,
    climatology: pd.Series,
) -> np.ndarray:
    doy, hour = doy_hour_keys(timestamps)
    keys = pd.MultiIndex.from_arrays([doy, hour], names=["DOY", "Hour"])
    return climatology.reindex(keys).to_numpy(dtype=float)


def fill_met_network(
    frame: pd.DataFrame,
    origins: dict[str, np.ndarray],
    stage0_observed: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    validation_rows: list[dict] = []
    method_rows: list[dict] = []
    dedicated = frame.loc[frame["Station"].isin(DEDICATED_MET_STATIONS)]

    for variable in MET_COMPLETION_VARIABLES:
        print(f"MET completion: {variable}", flush=True)
        wide = dedicated.pivot(index="Timestamp", columns="Station", values=variable).sort_index()
        wide = wide.reindex(columns=sorted(DEDICATED_MET_STATIONS))
        climatology, global_value = pooled_met_climatology(stage0_observed, variable)

        # Leave-one-station-out validation uses only original Stage 0 truth.
        all_truth: list[np.ndarray] = []
        all_predictions: list[np.ndarray] = []
        for station in sorted(DEDICATED_MET_STATIONS):
            truth_rows = stage0_observed.loc[
                (stage0_observed["Station"] == station) & stage0_observed[variable].notna(),
                ["Timestamp", variable],
            ]
            donor_values = wide.drop(columns=station).reindex(truth_rows["Timestamp"]).to_numpy(dtype=float)
            if variable == "Wind direction":
                prediction = circular_mean(donor_values)
            else:
                prediction = row_nanmedian(donor_values)
            station_climatology, station_global = pooled_met_climatology(
                stage0_observed, variable, excluded_station=station
            )
            fallback = lookup_climatology(truth_rows["Timestamp"], station_climatology)
            prediction = np.where(np.isfinite(prediction), prediction, fallback)
            prediction = np.where(np.isfinite(prediction), prediction, station_global)
            prediction = clip_new_values(variable, prediction)
            truth = truth_rows[variable].to_numpy(dtype=float)
            metrics = error_metrics(truth, prediction, circular=variable == "Wind direction")
            validation_rows.append({
                "Family": "MET",
                "Variable": variable,
                "Validation": "leave_one_station_out",
                "Station": station,
                **metrics,
            })
            all_truth.append(truth)
            all_predictions.append(prediction)

        overall = error_metrics(
            np.concatenate(all_truth),
            np.concatenate(all_predictions),
            circular=variable == "Wind direction",
        )
        validation_rows.append({
            "Family": "MET",
            "Variable": variable,
            "Validation": "leave_one_station_out",
            "Station": "ALL",
            **overall,
        })

        missing_positions = np.flatnonzero(origins[variable] == MODELING_COMPLETION)
        missing_times = frame.iloc[missing_positions]["Timestamp"]
        donor_matrix = wide.reindex(missing_times).to_numpy(dtype=float)
        if variable == "Wind direction":
            prediction = circular_mean(donor_matrix)
        else:
            prediction = row_nanmedian(donor_matrix)
        network_used = np.isfinite(prediction)
        fallback = lookup_climatology(missing_times, climatology)
        climatology_used = ~network_used & np.isfinite(fallback)
        prediction = np.where(network_used, prediction, fallback)
        global_used = ~np.isfinite(prediction)
        prediction[global_used] = global_value
        prediction = clip_new_values(variable, prediction)
        if not np.isfinite(prediction).all():
            raise ValueError(f"{variable}: MET completion produced non-finite values")
        frame.loc[frame.index[missing_positions], variable] = prediction
        method_rows.extend([
            {"Family": "MET", "Variable": variable, "Method": "concurrent_network", "Count": int(network_used.sum())},
            {"Family": "MET", "Variable": variable, "Method": "doy_hour_climatology", "Count": int(climatology_used.sum())},
            {"Family": "MET", "Variable": variable, "Method": "global_climatology", "Count": int(global_used.sum())},
        ])
    return validation_rows, method_rows


def soil_feature_spec(target: str) -> list[str]:
    if target in SOIL_MOISTURE:
        return [variable for variable in SOIL_MOISTURE if variable != target] + SOIL_TEMPERATURE + ["Ppt"]
    return [variable for variable in SOIL_TEMPERATURE if variable != target] + SOIL_MOISTURE + ["Ppt"]


def make_soil_features(
    frozen_values: dict[str, np.ndarray],
    target: str,
    positions: np.ndarray,
    station_codes: np.ndarray,
    time_features: np.ndarray,
) -> np.ndarray:
    columns = [station_codes[positions].astype(np.float32)]
    for variable in soil_feature_spec(target):
        columns.append(frozen_values[variable][positions])
    columns.extend(time_features[positions, column] for column in range(time_features.shape[1]))
    return np.column_stack(columns).astype(np.float32, copy=False)


def fit_soil_model(features: np.ndarray, target: np.ndarray) -> HistGradientBoostingRegressor:
    categorical = np.zeros(features.shape[1], dtype=bool)
    categorical[0] = True
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        categorical_features=categorical,
        random_state=RANDOM_SEED,
    )
    model.fit(features, target)
    return model


def select_block_holdouts(
    frame: pd.DataFrame,
    observed_positions: np.ndarray,
    holdout_station: str,
) -> np.ndarray:
    observed = frame.iloc[observed_positions][["Station", "Timestamp"]].copy()
    selected: list[np.ndarray] = []
    for station, group in observed.loc[observed["Station"] != holdout_station].groupby("Station", sort=True):
        times = pd.DatetimeIndex(group["Timestamp"]).sort_values()
        if len(times) < 168:
            continue
        anchor = times[int(len(times) * 0.6)]
        end = anchor + pd.Timedelta(hours=167)
        mask = (group["Timestamp"] >= anchor) & (group["Timestamp"] <= end)
        selected.append(group.index[mask].to_numpy(dtype=int))
    return np.sort(np.concatenate(selected)) if selected else np.array([], dtype=int)


def fill_soil_models(
    frame: pd.DataFrame,
    origins: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    validation_rows: list[dict] = []
    method_rows: list[dict] = []
    station_names = sorted(frame["Station"].unique())
    station_map = {station: code for code, station in enumerate(station_names)}
    station_codes = frame["Station"].map(station_map).to_numpy(dtype=np.int16)
    timestamps = pd.DatetimeIndex(frame["Timestamp"])
    hour_angle = 2.0 * np.pi * timestamps.hour.to_numpy() / 24.0
    doy_angle = 2.0 * np.pi * (timestamps.dayofyear.to_numpy() - 1) / 365.25
    time_features = np.column_stack([
        np.sin(hour_angle), np.cos(hour_angle), np.sin(doy_angle), np.cos(doy_angle)
    ]).astype(np.float32)
    frozen_values = {
        variable: frame[variable].to_numpy(dtype=np.float32, copy=True)
        for variable in [*SOIL_VARIABLES, "Ppt"]
    }

    for variable_number, variable in enumerate(SOIL_VARIABLES):
        started = time.monotonic()
        print(f"Soil completion: {variable}", flush=True)
        observed_positions = np.flatnonzero(origins[variable] == OBSERVED)
        missing_positions = np.flatnonzero(origins[variable] == MODELING_COMPLETION)
        if not len(observed_positions):
            raise ValueError(f"{variable}: no original observations available for pooled model")

        observed_station_counts = frame.iloc[observed_positions].groupby("Station").size()
        eligible = observed_station_counts.drop(labels=["CB15"], errors="ignore") if variable == "SWC_10" else observed_station_counts
        holdout_station = str(eligible.sort_values(ascending=False).index[0])
        station_holdout = observed_positions[
            frame["Station"].to_numpy()[observed_positions] == holdout_station
        ]
        block_holdout = select_block_holdouts(frame, observed_positions, holdout_station)
        excluded = np.union1d(station_holdout, block_holdout)
        validation_train = np.setdiff1d(observed_positions, excluded, assume_unique=True)
        validation_train = deterministic_sample(validation_train, MAX_TRAIN_ROWS, variable_number * 10 + 1)
        station_eval = deterministic_sample(station_holdout, MAX_VALIDATION_ROWS, variable_number * 10 + 2)
        block_eval = deterministic_sample(block_holdout, MAX_VALIDATION_ROWS, variable_number * 10 + 3)

        validation_model = fit_soil_model(
            make_soil_features(frozen_values, variable, validation_train, station_codes, time_features),
            frame[variable].to_numpy(dtype=float)[validation_train],
        )
        for validation_name, evaluation_positions, station in [
            ("missing_block", block_eval, "MULTIPLE"),
            ("missing_sensor", station_eval, holdout_station),
        ]:
            prediction = validation_model.predict(
                make_soil_features(frozen_values, variable, evaluation_positions, station_codes, time_features)
            )
            prediction = clip_new_values(variable, prediction)
            truth = frame[variable].to_numpy(dtype=float)[evaluation_positions]
            validation_rows.append({
                "Family": "Soil",
                "Variable": variable,
                "Validation": validation_name,
                "Station": station,
                **error_metrics(truth, prediction),
            })

        final_train = deterministic_sample(observed_positions, MAX_TRAIN_ROWS, variable_number * 10 + 4)
        final_model = fit_soil_model(
            make_soil_features(frozen_values, variable, final_train, station_codes, time_features),
            frame[variable].to_numpy(dtype=float)[final_train],
        )
        prediction = np.empty(len(missing_positions), dtype=float)
        for start in range(0, len(missing_positions), PREDICTION_CHUNK_ROWS):
            chunk = missing_positions[start : start + PREDICTION_CHUNK_ROWS]
            prediction[start : start + len(chunk)] = final_model.predict(
                make_soil_features(frozen_values, variable, chunk, station_codes, time_features)
            )
        prediction = clip_new_values(variable, prediction)

        # This fallback should be unreachable for HGB, but keeps the complete-
        # matrix contract explicit if a future estimator changes.
        nonfinite = ~np.isfinite(prediction)
        climatology_count = 0
        global_count = 0
        if nonfinite.any():
            observed_times = timestamps[observed_positions]
            observed_values = frame[variable].to_numpy(dtype=float)[observed_positions]
            climate_data = pd.DataFrame({
                "DOY": observed_times.dayofyear,
                "Hour": observed_times.hour,
                "Value": observed_values,
            })
            climatology = climate_data.groupby(["DOY", "Hour"])["Value"].median()
            fallback = lookup_climatology(timestamps[missing_positions[nonfinite]], climatology)
            use_climatology = np.isfinite(fallback)
            affected = np.flatnonzero(nonfinite)
            prediction[affected[use_climatology]] = fallback[use_climatology]
            climatology_count = int(use_climatology.sum())
            still_nonfinite = ~np.isfinite(prediction)
            global_count = int(still_nonfinite.sum())
            prediction[still_nonfinite] = float(np.nanmedian(observed_values))
            prediction = clip_new_values(variable, prediction)
        if not np.isfinite(prediction).all():
            raise ValueError(f"{variable}: Soil completion produced non-finite values")
        frame.loc[frame.index[missing_positions], variable] = prediction
        method_rows.extend([
            {"Family": "Soil", "Variable": variable, "Method": "pooled_hist_gradient_boosting", "Count": int(len(missing_positions) - climatology_count - global_count)},
            {"Family": "Soil", "Variable": variable, "Method": "pooled_doy_hour_climatology", "Count": climatology_count},
            {"Family": "Soil", "Variable": variable, "Method": "global_observed_median", "Count": global_count},
        ])
        print(f"  filled={len(missing_positions):,}; runtime={time.monotonic() - started:.1f}s", flush=True)
    return validation_rows, method_rows


def build_quality_summary(
    frame: pd.DataFrame,
    origins: dict[str, np.ndarray],
    source_absent: set[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for station, positions in frame.groupby("Station", sort=True).indices.items():
        positions = np.asarray(positions)
        total = len(positions)
        for variable in MEASUREMENT_VARIABLES:
            codes = origins[variable][positions]
            observed = int((codes == OBSERVED).sum())
            pipeline = int((codes == PIPELINE_IMPUTED).sum())
            modeling = int((codes == MODELING_COMPLETION).sum())
            if observed + pipeline + modeling != total:
                raise ValueError(f"Origin counts do not reconcile for {station} {variable}")
            rows.append({
                "Station": station,
                "Variable": variable,
                "Total_Count": total,
                "Observed_Count": observed,
                "Observed_Percent": observed / total * 100.0,
                "Pipeline_Imputed_Count": pipeline,
                "Pipeline_Imputed_Percent": pipeline / total * 100.0,
                "Modeling_Completion_Count": modeling,
                "Modeling_Completion_Percent": modeling / total * 100.0,
                "Dedicated_MET": station in DEDICATED_MET_STATIONS,
                "Source_Sensor_Absent": (station, variable) in source_absent,
                "Pending_QC": (station, variable) in PENDING_QC,
            })
    summary = pd.DataFrame(rows)
    percent_total = summary[
        ["Observed_Percent", "Pipeline_Imputed_Percent", "Modeling_Completion_Percent"]
    ].sum(axis=1)
    if not np.allclose(percent_total, 100.0, atol=1e-10):
        raise ValueError("Quality-summary percentages do not reconcile to 100%")
    return summary


def station_quality(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby("Station", sort=True)[
        ["Total_Count", "Observed_Count", "Pipeline_Imputed_Count", "Modeling_Completion_Count"]
    ].sum()
    for prefix, count_column in [
        ("Observed", "Observed_Count"),
        ("Pipeline_Imputed", "Pipeline_Imputed_Count"),
        ("Modeling_Completion", "Modeling_Completion_Count"),
    ]:
        grouped[f"{prefix}_Percent"] = grouped[count_column] / grouped["Total_Count"] * 100.0
    return grouped.reset_index()


def format_metric(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def build_readme(
    frame: pd.DataFrame,
    quality_summary: pd.DataFrame,
    validation: pd.DataFrame,
    source_absent: set[tuple[str, str]],
) -> str:
    station_summary = station_quality(quality_summary)
    top = station_summary.nlargest(5, "Observed_Percent")["Station"].tolist()
    bottom = station_summary.nsmallest(5, "Observed_Percent")["Station"].tolist()
    return f"""# Complete TxSON33 Dataset for Modeling

## Overview

The original TxSON imputation pipeline fills valid gaps within each sensor's
observed coverage. It intentionally preserves structural missingness when a
sensor was never installed, a timestamp falls outside source coverage, or a
station has no dedicated MET measurements.

Modeling workflows need a complete feature matrix across all 33 stations. This
dataset starts from the base imputation output and applies a separate,
modeling-only completion step to the remaining missing values.

No existing finite value in the base output is changed. Additional values are
labeled `modeling_completion` so they remain distinguishable from observations
and values filled by the original imputation pipeline.

## Quick Start

Use `TxSON33_modeling_complete.parquet` for Python analysis:

```python
import pandas as pd

df = pd.read_parquet("TxSON33_modeling_complete.parquet")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
```

The CSV contains the same rows and columns and is provided as a portable format:

```python
df = pd.read_csv("TxSON33_modeling_complete.csv", parse_dates=["Timestamp"])
```

## Dataset Contents

- Stations: **{frame['Station'].nunique()}**
- Rows: **{len(frame):,}**
- Overall date range: **{frame['Timestamp'].min()}** to **{frame['Timestamp'].max()}**
- Required measurement variables: **{len(MEASUREMENT_VARIABLES)}**
- Missing required measurement values: **0**
- Unique key: `Station` + `Timestamp`

Measurement columns:

- Soil moisture: `SWC_5`, `SWC_10`, `SWC_20`, `SWC_50`
- Soil temperature: `T_5`, `T_10`, `T_20`, `T_50`
- Weather: `Tair`, `RH`, `Srad`, `Wind speed`, `Wind direction`, `Ppt`

## How Remaining Missing Values Were Completed

The modeling-only step fills only values that were still missing in the base
imputation output.

- Soil variables use deterministic pooled gradient-boosting models with other
  Soil depths, `Ppt`, station identity, hour, and day-of-year features.
- MET variables use concurrent network information from the six dedicated-MET
  stations, followed by time-based climatology when concurrent data are absent.
- Wind direction is treated circularly so angles near north remain close.
- `Ppt` was already complete and is unchanged.

## Origin Columns

Every measurement has a matching `*_Origin` column:

- `observed`: unchanged from a finite source observation.
- `pipeline_imputed`: filled or replaced by the original imputation pipeline.
- `modeling_completion`: added only to complete the modeling matrix.

`Any_Soil_Imputed` and `Any_MET_Imputed` retain the original pipeline's row-level
flags. `Any_Modeling_Completion` identifies rows where the modeling-only step
added at least one value.

For evaluation, filter on the prediction target's own `*_Origin` column rather
than using a row-level flag.

## Data Coverage

Dedicated non-Ppt MET measurements exist at `CB01`, `CB04`, `CB06`, `FD02`,
`FD03`, and `WC05`. The other 27 stations use regional estimates for non-Ppt
MET variables.

`CB07`, `CB26`, `FD03`, `FD18`, `FD21`, and `FD24` originally lacked both
`SWC_50` and `T_50`.

- Highest original-data coverage: `{', '.join(top)}`.
- Lowest original-data coverage: `{', '.join(bottom)}`.

See `data_quality_summary.csv` for observed, pipeline-imputed, and
modeling-completion counts and percentages for every station-variable pair.

## Key Notes

- `modeling_completion` values are synthetic estimates and should not normally be
  used as validation or test ground truth for that variable.
- Only six stations have dedicated non-Ppt MET; the other 27 use regional
  estimates.
- `RH` and `Wind direction` have higher completion uncertainty than the other
  MET variables.
"""


def attach_origin_columns(frame: pd.DataFrame, origins: dict[str, np.ndarray]) -> pd.DataFrame:
    for variable in MEASUREMENT_VARIABLES:
        frame[f"{variable}_Origin"] = pd.Categorical.from_codes(
            origins[variable], categories=ORIGIN_LABELS
        )
    frame["Any_Modeling_Completion"] = np.column_stack([
        origins[variable] == MODELING_COMPLETION for variable in MEASUREMENT_VARIABLES
    ]).any(axis=1)
    return frame


def verify_complete_frame(
    complete: pd.DataFrame,
    origins: dict[str, np.ndarray],
    expected_rows: int,
) -> dict:
    if len(complete) != expected_rows:
        raise ValueError(f"Row count changed: expected {expected_rows}, found {len(complete)}")
    if complete["Station"].nunique() != 33:
        raise ValueError("Complete dataset does not contain exactly 33 stations")
    if complete.duplicated(["Station", "Timestamp"]).any():
        raise ValueError("Complete dataset has duplicate Station/Timestamp rows")
    monotonic = complete.groupby("Station", sort=False)["Timestamp"].apply(lambda values: values.is_monotonic_increasing)
    if not monotonic.all():
        raise ValueError(f"Non-monotonic station timelines: {monotonic.index[~monotonic].tolist()}")
    missing = complete[MEASUREMENT_VARIABLES].isna().sum()
    if missing.any():
        raise ValueError(f"Required values remain missing: {missing[missing > 0].to_dict()}")

    authoritative = pd.read_parquet(AUTHORITATIVE_FILE, columns=["Station", "Timestamp", *MEASUREMENT_VARIABLES])
    if not complete[["Station", "Timestamp"]].reset_index(drop=True).equals(
        authoritative[["Station", "Timestamp"]].reset_index(drop=True)
    ):
        raise ValueError("Station/Timestamp rows differ from the authoritative dataset")
    for variable in MEASUREMENT_VARIABLES:
        original = authoritative[variable].to_numpy(dtype=float)
        current = complete[variable].to_numpy(dtype=float)
        finite = np.isfinite(original)
        if not np.array_equal(original[finite], current[finite]):
            raise ValueError(f"A finite authoritative {variable} value changed")
        created = origins[variable] == MODELING_COMPLETION
        if not np.isfinite(current[created]).all():
            raise ValueError(f"{variable}: a modeling-completion value is non-finite")
        if variable == "Wind direction":
            if ((current[created] < 0.0) | (current[created] >= 360.0)).any():
                raise ValueError("New Wind direction values are outside [0, 360)")
        else:
            lower, upper = NEW_VALUE_RANGES[variable]
            if ((current[created] < lower) | (current[created] > upper)).any():
                raise ValueError(f"New {variable} values are outside [{lower}, {upper}]")
    return {
        "stations": int(complete["Station"].nunique()),
        "rows": int(len(complete)),
        "missing_required_values": int(complete[MEASUREMENT_VARIABLES].isna().sum().sum()),
        "duplicate_station_timestamps": int(complete.duplicated(["Station", "Timestamp"]).sum()),
        "authoritative_finite_values_changed": 0,
    }


def verify_written_outputs(temp_dir: Path, complete: pd.DataFrame, quality: pd.DataFrame) -> dict:
    parquet_path = temp_dir / "TxSON33_modeling_complete.parquet"
    csv_path = temp_dir / "TxSON33_modeling_complete.csv"
    reloaded = pd.read_parquet(parquet_path)
    if not reloaded.equals(complete):
        raise ValueError("Parquet round-trip differs from the completed in-memory dataset")
    del reloaded

    # Verify the larger CSV incrementally to avoid a second full-frame copy.
    offset = 0
    csv_max_absolute_difference = 0.0
    for chunk in pd.read_csv(csv_path, parse_dates=["Timestamp"], chunksize=200_000):
        expected = complete.iloc[offset : offset + len(chunk)].copy()
        for variable in MEASUREMENT_VARIABLES:
            actual_values = chunk[variable].to_numpy(dtype=float)
            expected_values = expected[variable].to_numpy(dtype=float)
            difference = np.abs(actual_values - expected_values)
            csv_max_absolute_difference = max(
                csv_max_absolute_difference, float(np.nanmax(difference, initial=0.0))
            )
            if not np.allclose(actual_values, expected_values, rtol=0.0, atol=1e-12):
                raise ValueError(f"CSV round-trip changed {variable} near row {offset}")
            origin_column = f"{variable}_Origin"
            if not np.array_equal(
                chunk[origin_column].to_numpy(dtype=str),
                expected[origin_column].astype(str).to_numpy(),
            ):
                raise ValueError(f"CSV round-trip changed {origin_column} near row {offset}")
        if not chunk[["Station", "Timestamp"]].reset_index(drop=True).equals(
            expected[["Station", "Timestamp"]].reset_index(drop=True)
        ):
            raise ValueError(f"CSV Station/Timestamp mismatch near row {offset}")
        offset += len(chunk)
    if offset != len(complete):
        raise ValueError(f"CSV row count mismatch: expected {len(complete)}, found {offset}")

    saved_quality = pd.read_csv(temp_dir / "data_quality_summary.csv")
    count_columns = ["Observed_Count", "Pipeline_Imputed_Count", "Modeling_Completion_Count"]
    if not saved_quality[count_columns].equals(quality[count_columns]):
        raise ValueError("Saved quality-summary counts do not match the computed summary")
    return {
        "parquet_round_trip": "passed",
        "csv_round_trip": "passed",
        "csv_max_absolute_float_round_trip_difference": csv_max_absolute_difference,
        "quality_summary_reconciliation": "passed",
    }


def build(overwrite: bool = False) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUT_DIR.parent / f".{OUTPUT_DIR.name}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if OUTPUT_DIR.exists() and not overwrite:
        raise FileExistsError(f"{OUTPUT_DIR} already exists; use --overwrite to replace it")
    temp_dir.mkdir(parents=True)

    source_snapshot = directory_snapshot([AUTHORITATIVE_DIR, STAGE0_DIR])
    authoritative_hash = sha256(AUTHORITATIVE_FILE)
    frame, origins, met_observed, missing_rows, source_absent = load_inputs()
    expected_rows = len(frame)
    if expected_rows != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} authoritative rows, found {expected_rows}")
    missing_structure = validate_known_structure(frame, missing_rows, source_absent)
    missing_structure.to_csv(REVIEW_DIR / "missing_structure.csv", index=False)
    print(
        missing_structure.groupby("Category")["Missing_Count"].sum().sort_index().to_string(),
        flush=True,
    )

    validation_rows, method_rows = fill_met_network(frame, origins, met_observed)
    soil_validation, soil_methods = fill_soil_models(frame, origins)
    validation_rows.extend(soil_validation)
    method_rows.extend(soil_methods)
    validation = pd.DataFrame(validation_rows)
    methods = pd.DataFrame(method_rows)
    validation.to_csv(REVIEW_DIR / "validation_metrics.csv", index=False)
    methods.to_csv(REVIEW_DIR / "completion_method_counts.csv", index=False)

    quality = build_quality_summary(frame, origins, source_absent)
    frame = attach_origin_columns(frame, origins)
    verification = verify_complete_frame(frame, origins, expected_rows)

    frame.to_parquet(temp_dir / "TxSON33_modeling_complete.parquet", index=False, compression="zstd")
    frame.to_csv(temp_dir / "TxSON33_modeling_complete.csv", index=False)
    quality.to_csv(temp_dir / "data_quality_summary.csv", index=False)
    (temp_dir / "README.md").write_text(
        build_readme(frame, quality, validation, source_absent), encoding="utf-8"
    )
    verification.update(verify_written_outputs(temp_dir, frame, quality))

    expected_files = {
        "TxSON33_modeling_complete.parquet",
        "TxSON33_modeling_complete.csv",
        "data_quality_summary.csv",
        "README.md",
    }
    actual_files = {path.name for path in temp_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(f"Unexpected final file set: {sorted(actual_files)}")
    if directory_snapshot([AUTHORITATIVE_DIR, STAGE0_DIR]) != source_snapshot:
        raise ValueError("A scientific/source input file changed during the build")
    if sha256(AUTHORITATIVE_FILE) != authoritative_hash:
        raise ValueError("The authoritative Parquet content changed during the build")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    temp_dir.rename(OUTPUT_DIR)
    verification.update({
        "authoritative_sha256_before_after": authoritative_hash,
        "source_snapshots_unchanged": True,
        "output_files": sorted(expected_files),
    })
    (REVIEW_DIR / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing derived output directory")
    args = parser.parse_args()
    build(overwrite=args.overwrite)


if __name__ == "__main__":
    main()

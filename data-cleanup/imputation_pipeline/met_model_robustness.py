"""Multi-seed artificial-gap robustness benchmark for non-Ppt MET data."""

from __future__ import annotations

import argparse
import hashlib
import signal
import time
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from MetGaps import (
    BOUNDARY_REVIEW_LIMITS,
    MET_GAP_METHODS,
    NON_PPT_MET_PARAMS,
    PHYSICAL_BOUNDS,
    apply_bounds,
    build_model_features,
    donor_regression_prediction,
    fill_sparse_prediction_holes,
    linear_run_prediction,
    sample_training_rows,
    tree_prediction,
)


BENCHMARK_VERSION = "2026-08-08-met-robustness-v3"
TARGETED_VERSION = "2026-08-08-met-wind-verylong-confirmation-v1"
DEPLOYMENT_VERSION = "2026-08-08-met-wind-deployment-confirmation-v1"
COVERAGE_VERSION = "2026-08-08-met-deployment-coverage-v1"
DEFAULT_SEEDS = (42, 7, 21, 84)
TARGETED_SEED = 126
DEPLOYMENT_SEED = 127
COVERAGE_SEED = 128
TARGETED_PARAMETER = "Wind speed"
TARGETED_GAP_CLASS = "verylong"
TARGETED_METHODS = ("donor_regression", "xgboost")
TARGETED_SEASONS = ("DJF", "MAM", "JJA", "SON")
TARGETED_MIN_RELATIVE_IMPROVEMENT = 0.05
TARGETED_MAX_PHYSICAL_VIOLATION_RATE = 0.001
GAP_CLASSES = ("short", "medium", "long", "verylong")
GAP_LENGTH_RANGES = {
    "short": (6, 23),
    "medium": (24, 167),
    "long": (168, 719),
    "verylong": (720, 1_440),
}
GAPS_PER_STATION = 5
STANDARD_METHODS = (
    "linear",
    "seasonal_hour",
    "donor_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
)
SOURCE_STATIONS = ("CB01", "CB04", "CB06", "FD02", "FD03", "WC05")
SARIMAX_TIMEOUT_SECONDS = 60

# Freeze the production map that the v3 robustness benchmark evaluated. This
# keeps the candidate-selection record reproducible after a confirmed change.
ROBUSTNESS_BASELINE_METHODS = {
    parameter: gap_methods.copy()
    for parameter, gap_methods in MET_GAP_METHODS.items()
}
ROBUSTNESS_BASELINE_METHODS["Wind speed"]["verylong"] = "donor_regression"

METHOD_LABELS = {
    "linear": "Linear",
    "seasonal_hour": "Seasonal hour",
    "donor_regression": "Donor regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "xgboost": "XGBoost",
    "sarimax": "SARIMAX",
}
METHOD_SHORT_LABELS = {
    "linear": "Linear",
    "seasonal_hour": "Seasonal",
    "donor_regression": "Donor",
    "random_forest": "RF",
    "hist_gradient_boosting": "HGB",
    "xgboost": "XGB",
}
METHOD_COLORS = {
    "linear": "#4C78A8",
    "seasonal_hour": "#F2A541",
    "donor_regression": "#3A8D5D",
    "random_forest": "#A05195",
    "hist_gradient_boosting": "#54A8B6",
    "xgboost": "#D95F59",
}


def pipeline_dir() -> Path:
    candidates = (
        Path.cwd(),
        Path.cwd() / "data-cleanup" / "imputation_pipeline",
        Path(__file__).resolve().parent,
    )
    for candidate in candidates:
        if (candidate / "cleaned_data").is_dir() and (candidate / "MetGaps.py").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate data-cleanup/imputation_pipeline")


def stable_seed(*parts: object) -> int:
    label = "|".join(map(str, parts))
    return int.from_bytes(
        hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest(),
        "little",
    )


def season_for(index: pd.DatetimeIndex) -> str:
    month = index[len(index) // 2].month
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def load_qc_masked_frames(
    base_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load Stage 0 observations and remove confirmed MET sensor-QC periods."""
    base_dir = base_dir or pipeline_dir()
    frames: dict[str, pd.DataFrame] = {}
    for station in SOURCE_STATIONS:
        path = base_dir / "cleaned_data" / f"Station{station}_cleaned_data.csv"
        frame = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
        frame.index = pd.DatetimeIndex(frame.index)
        frame = frame[~frame.index.duplicated(keep="first")].sort_index()
        frame = frame.reindex(
            pd.date_range(frame.index.min(), frame.index.max(), freq="h", name="Date")
        )
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        # Use the reconciled direct-source Ppt series as the local rainfall
        # driver, matching the approved MET-first source policy.
        met_path = (
            base_dir
            / "met_output"
            / f"Station{station}_met_filled_shortgaps.csv"
        )
        if met_path.exists():
            met = pd.read_csv(
                met_path,
                index_col="Date",
                parse_dates=True,
                low_memory=False,
            )
            if "Ppt" in met:
                frame["Ppt"] = pd.to_numeric(
                    met["Ppt"], errors="coerce"
                ).reindex(frame.index)
        frames[station] = frame

    mask_path = (
        base_dir
        / "met_qc_reports"
        / "model_fill"
        / "met_sensor_qc_masked_segments.csv"
    )
    masks = pd.read_csv(mask_path, parse_dates=["Start", "End"])
    audit_rows = []
    for _, row in masks.iterrows():
        station = str(row["Station"])
        parameter = str(row["Parameter"])
        if station not in frames or parameter not in frames[station]:
            continue
        start = pd.Timestamp(row["Start"])
        end = pd.Timestamp(row["End"])
        selected = frames[station].loc[start:end, parameter]
        observed = int(selected.notna().sum())
        frames[station].loc[start:end, parameter] = np.nan
        audit_rows.append(
            {
                "Station": station,
                "Parameter": parameter,
                "Start": start,
                "End": end,
                "Reported Masked Hours": int(row["Masked Hours"]),
                "Observed Hours Excluded": observed,
                "Rule": row["Rule"],
            }
        )
    audit = pd.DataFrame(audit_rows)
    return frames, audit


def observed_runs(series: pd.Series) -> list[tuple[int, int]]:
    valid = series.notna().to_numpy()
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def sample_parameter_cases(
    series: pd.Series,
    station: str,
    parameter: str,
    seed: int,
    gaps_per_class: int = GAPS_PER_STATION,
) -> list[dict]:
    """Select non-overlapping bracketed gaps, largest classes first."""
    rng = np.random.default_rng(stable_seed(BENCHMARK_VERSION, seed, station, parameter))
    runs = observed_runs(series)
    occupied = np.zeros(len(series), dtype=bool)
    rows = []
    for gap_class in ("verylong", "long", "medium", "short"):
        low, high = GAP_LENGTH_RANGES[gap_class]
        candidates = [run for run in runs if run[1] - run[0] + 1 >= low + 2]
        for sample_number in range(1, gaps_per_class + 1):
            selected = None
            for _ in range(4_000):
                if not candidates:
                    break
                run_start, run_end = candidates[int(rng.integers(0, len(candidates)))]
                max_length = min(high, run_end - run_start - 1)
                if max_length < low:
                    continue
                length = int(rng.integers(low, max_length + 1))
                start = int(rng.integers(run_start + 1, run_end - length + 1))
                end = start + length - 1
                buffer_start = max(0, start - 1)
                buffer_end = min(len(series), end + 2)
                if occupied[buffer_start:buffer_end].any():
                    continue
                occupied[start:end + 1] = True
                selected = pd.DatetimeIndex(series.index[start:end + 1])
                break
            if selected is None:
                raise RuntimeError(
                    f"Could not sample {gap_class} case {sample_number} for "
                    f"{station} {parameter} seed {seed}"
                )
            case_id = (
                f"{seed}|{station}|{parameter}|{gap_class}|{sample_number}|"
                f"{selected[0].isoformat()}"
            )
            rows.append(
                {
                    "Benchmark Version": BENCHMARK_VERSION,
                    "Seed": seed,
                    "Case ID": case_id,
                    "Station": station,
                    "Parameter": parameter,
                    "Gap Class": gap_class,
                    "Sample": sample_number,
                    "Start": selected[0],
                    "End": selected[-1],
                    "Hours": len(selected),
                    "Season": season_for(selected),
                }
            )
    return rows


def sample_seed_cases(
    frames: dict[str, pd.DataFrame],
    seed: int,
) -> pd.DataFrame:
    rows = []
    for station in SOURCE_STATIONS:
        for parameter in NON_PPT_MET_PARAMS:
            source = frames[station][parameter]
            if source.notna().sum() < 500:
                raise RuntimeError(f"Insufficient observations for {station} {parameter}")
            rows.extend(sample_parameter_cases(source, station, parameter, seed))
    return pd.DataFrame(rows).sort_values(
        ["Station", "Parameter", "Gap Class", "Sample"]
    ).reset_index(drop=True)


def circular_difference(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs((predicted - actual + 180.0) % 360.0 - 180.0)


def seasonal_hour_prediction(
    masked: pd.Series,
    predict_index: pd.DatetimeIndex,
    circular: bool,
) -> pd.Series:
    training = masked.dropna()
    keys = [training.index.month, training.index.hour]
    if circular:
        radians = np.deg2rad(training.to_numpy(dtype=float))
        sin_mean = pd.Series(np.sin(radians), index=training.index).groupby(keys).mean()
        cos_mean = pd.Series(np.cos(radians), index=training.index).groupby(keys).mean()
        fallback_sin = float(np.sin(radians).mean())
        fallback_cos = float(np.cos(radians).mean())
        values = []
        for timestamp in predict_index:
            key = (timestamp.month, timestamp.hour)
            values.append(
                (np.degrees(np.arctan2(
                    sin_mean.get(key, fallback_sin),
                    cos_mean.get(key, fallback_cos),
                )) + 360.0) % 360.0
            )
        return pd.Series(values, index=predict_index)
    grouped = training.groupby(keys).mean()
    fallback = float(training.mean())
    return pd.Series(
        [grouped.get((ts.month, ts.hour), fallback) for ts in predict_index],
        index=predict_index,
    )


def hist_gradient_prediction(
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

    def model(seed: int):
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.06,
                max_leaf_nodes=31,
                random_state=seed,
            ),
        )

    if parameter == "Wind direction":
        radians = np.deg2rad(train_y.to_numpy(dtype=float))
        sin_model = model(42)
        cos_model = model(43)
        sin_model.fit(train_x, np.sin(radians))
        cos_model.fit(train_x, np.cos(radians))
        values = (
            np.degrees(np.arctan2(
                sin_model.predict(test_x),
                cos_model.predict(test_x),
            )) + 360.0
        ) % 360.0
        return pd.Series(values, index=predict_index)
    fitted = model(42)
    fitted.fit(train_x, train_y)
    return pd.Series(fitted.predict(test_x), index=predict_index)


def predict_method(
    method: str,
    station: str,
    parameter: str,
    masked: pd.Series,
    cases: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    seed: int,
    benchmark_version: str = BENCHMARK_VERSION,
) -> pd.Series:
    indexes = {
        row["Case ID"]: pd.date_range(row.Start, row.End, freq="h")
        for _, row in cases.iterrows()
    }
    predict_index = pd.DatetimeIndex(
        sorted({timestamp for index in indexes.values() for timestamp in index})
    )
    if method == "linear":
        pieces = [
            linear_run_prediction(masked, index, parameter)
            for index in indexes.values()
        ]
        return pd.concat(pieces).sort_index()
    if method == "seasonal_hour":
        return seasonal_hour_prediction(
            masked, predict_index, parameter == "Wind direction"
        )

    features = build_model_features(station, parameter, masked, frames)
    rng = np.random.default_rng(
        stable_seed(benchmark_version, seed, station, parameter, "training")
    )
    if method == "donor_regression":
        prediction = donor_regression_prediction(
            features, masked, predict_index
        )
        return fill_sparse_prediction_holes(prediction, parameter)
    if method in {"random_forest", "xgboost"}:
        return tree_prediction(
            method, features, masked, predict_index, parameter, rng
        )
    if method == "hist_gradient_boosting":
        return hist_gradient_prediction(
            features, masked, predict_index, parameter, rng
        )
    raise ValueError(method)


def score_case(
    case: pd.Series,
    source: pd.Series,
    masked: pd.Series,
    prediction: pd.Series,
) -> dict:
    index = pd.date_range(case.Start, case.End, freq="h")
    truth = source.reindex(index)
    raw = pd.to_numeric(prediction.reindex(index), errors="coerce")
    valid = truth.notna() & raw.notna() & np.isfinite(raw)
    if not valid.all():
        raise ValueError(f"prediction missing {int((~valid).sum())} test hours")

    lower, upper = PHYSICAL_BOUNDS[case.Parameter]
    violations = int((raw < lower).sum())
    if upper is not None:
        violations += int((raw > upper).sum())
    predicted = apply_bounds(raw, case.Parameter)
    actual_values = truth.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    if case.Parameter == "Wind direction":
        errors = circular_difference(actual_values, predicted_values)
        residuals = errors
        scale = 180.0
    else:
        residuals = predicted_values - actual_values
        errors = np.abs(residuals)
        observed = source.dropna()
        scale = float(observed.quantile(0.75) - observed.quantile(0.25))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(observed.std())
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    left_ts = index[0] - pd.Timedelta(hours=1)
    right_ts = index[-1] + pd.Timedelta(hours=1)
    boundary_values = []
    for observed_value, predicted_value in (
        (masked.get(left_ts, np.nan), predicted.iloc[0]),
        (masked.get(right_ts, np.nan), predicted.iloc[-1]),
    ):
        if pd.isna(observed_value):
            boundary_values.append(np.nan)
        elif case.Parameter == "Wind direction":
            boundary_values.append(float(circular_difference(
                np.array([observed_value]), np.array([predicted_value])
            )[0]))
        else:
            boundary_values.append(float(abs(observed_value - predicted_value)))

    if case.Parameter == "Wind direction":
        hourly = circular_difference(
            predicted_values[:-1], predicted_values[1:]
        )
    else:
        hourly = np.abs(np.diff(predicted_values))
    return {
        "MAE": float(errors.mean()),
        "RMSE": rmse,
        "Normalized MAE": float(errors.mean() / scale),
        "Normalized RMSE": float(rmse / scale),
        "Scale": scale,
        "Bias": (
            np.nan
            if case.Parameter == "Wind direction"
            else float(residuals.mean())
        ),
        "Physical Violations": violations,
        "Left Boundary Jump": boundary_values[0],
        "Right Boundary Jump": boundary_values[1],
        "Max Hourly Change": float(hourly.max()) if len(hourly) else np.nan,
        "Boundary Screen Exceeded": bool(
            any(
                np.isfinite(value)
                and value > BOUNDARY_REVIEW_LIMITS[case.Parameter]
                for value in boundary_values
            )
        ),
    }


def methods_for(parameter: str) -> tuple[str, ...]:
    if parameter == "Wind direction":
        return tuple(method for method in STANDARD_METHODS if method != "donor_regression")
    return STANDARD_METHODS


def run_seed(
    frames: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    seed: int,
    detail_path: Path,
    selected_methods: Iterable[str] | None = None,
    benchmark_version: str = BENCHMARK_VERSION,
) -> pd.DataFrame:
    rows: list[dict] = []
    if detail_path.exists():
        cached = pd.read_csv(detail_path, parse_dates=["Start", "End"])
        if (
            "Benchmark Version" in cached
            and cached["Benchmark Version"].eq(benchmark_version).all()
        ):
            rows = cached.to_dict("records")
            print(f"Seed {seed}: reused {len(rows)} checkpoint rows")

    for (station, parameter), group in cases.groupby(["Station", "Parameter"]):
        source = frames[station][parameter].copy()
        all_index = pd.DatetimeIndex(
            sorted({
                timestamp
                for row in group.itertuples()
                for timestamp in pd.date_range(row.Start, row.End, freq="h")
            })
        )
        masked = source.copy()
        masked.loc[all_index] = np.nan
        expected_ids = set(group["Case ID"])
        methods = tuple(selected_methods) if selected_methods else methods_for(parameter)
        for method in methods:
            existing_ids = {
                row["Case ID"]
                for row in rows
                if row["Station"] == station
                and row["Parameter"] == parameter
                and row["Method"] == method
            }
            if existing_ids == expected_ids:
                continue
            rows = [
                row for row in rows
                if not (
                    row["Station"] == station
                    and row["Parameter"] == parameter
                    and row["Method"] == method
                )
            ]
            started = time.perf_counter()
            error = ""
            try:
                predictions = predict_method(
                    method, station, parameter, masked, group, frames, seed,
                    benchmark_version=benchmark_version,
                )
            except Exception as exc:
                predictions = pd.Series(np.nan, index=all_index)
                error = str(exc)
            runtime = time.perf_counter() - started
            for case_number, (_, case) in enumerate(group.iterrows()):
                base = {
                    **case.to_dict(),
                    "Method": method,
                    "Runtime Seconds": runtime if case_number == 0 else 0.0,
                }
                try:
                    if error:
                        raise RuntimeError(error)
                    metrics = score_case(case, source, masked, predictions)
                    rows.append({**base, **metrics, "Status": "ok", "Error": ""})
                except Exception as exc:
                    rows.append({**base, "Status": "failed", "Error": str(exc)})
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(detail_path, index=False)
            print(
                f"Seed {seed}: {station} {parameter} {method} "
                f"({runtime:.1f}s)"
            )
    return pd.DataFrame(rows)


def sarimax_prediction(
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
) -> pd.Series:
    context_start = gap_index[0] - pd.Timedelta(days=90)
    context_end = gap_index[-1] + pd.Timedelta(days=30)
    local = masked.loc[
        max(masked.index.min(), context_start):min(masked.index.max(), context_end)
    ]
    if local.notna().sum() < 24 * 14:
        raise ValueError("insufficient SARIMAX context")

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"SARIMAX exceeded {SARIMAX_TIMEOUT_SECONDS}s")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, SARIMAX_TIMEOUT_SECONDS)
    try:
        model = SARIMAX(
            local,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 0, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted = model.fit(disp=False, maxiter=60)
        if not bool(fitted.mle_retvals.get("converged", True)):
            raise RuntimeError("SARIMAX did not converge")
        return fitted.get_prediction(
            start=gap_index[0], end=gap_index[-1]
        ).predicted_mean.reindex(gap_index)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_sarimax_subset(
    frames: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    selected = (
        cases[
            cases["Gap Class"].eq("medium")
            & ~cases["Parameter"].eq("Wind direction")
        ]
        .sort_values(["Seed", "Station", "Parameter", "Hours", "Case ID"])
        .groupby(["Seed", "Station", "Parameter"], as_index=False)
        .first()
    )
    rows = []
    if output_path.exists():
        cached = pd.read_csv(output_path, parse_dates=["Start", "End"])
        if cached["Benchmark Version"].eq(BENCHMARK_VERSION).all():
            rows = cached.to_dict("records")
    completed = {row["Case ID"] for row in rows}
    for _, case in selected.iterrows():
        if case["Case ID"] in completed:
            continue
        source = frames[case.Station][case.Parameter].copy()
        index = pd.date_range(case.Start, case.End, freq="h")
        masked = source.copy()
        masked.loc[index] = np.nan
        started = time.perf_counter()
        try:
            prediction = sarimax_prediction(masked, index)
            metrics = score_case(case, source, masked, prediction)
            row = {
                **case.to_dict(),
                "Method": "sarimax",
                **metrics,
                "Runtime Seconds": time.perf_counter() - started,
                "Status": "ok",
                "Error": "",
            }
        except Exception as exc:
            row = {
                **case.to_dict(),
                "Method": "sarimax",
                "Runtime Seconds": time.perf_counter() - started,
                "Status": "failed",
                "Error": str(exc),
            }
        rows.append(row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(
            f"SARIMAX: seed {case.Seed} {case.Station} {case.Parameter} "
            f"{row['Status']} ({row['Runtime Seconds']:.1f}s)"
        )
    return pd.DataFrame(rows)


def aggregate_model_scores(
    cases: pd.DataFrame,
    detail: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    planned = cases.groupby(group_columns).size().rename("Planned Cases")
    rows = []
    for keys, group in detail.groupby(group_columns + ["Method"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns + ["Method"], keys))
        plan_key = tuple(key_values[column] for column in group_columns)
        if len(group_columns) == 1:
            plan_key = plan_key[0]
        successful = group[group["Status"].eq("ok")]
        rows.append(
            {
                **key_values,
                "Planned Cases": int(planned.loc[plan_key]),
                "Successful Cases": int(len(successful)),
                "Failed Cases": int((group["Status"] != "ok").sum()),
                "Mean Normalized RMSE": successful["Normalized RMSE"].mean(),
                "Mean Normalized MAE": successful["Normalized MAE"].mean(),
                "Mean MAE": successful["MAE"].mean(),
                "Predicted Hours": int(successful["Hours"].fillna(0).sum()),
                "Physical Violations": int(
                    successful["Physical Violations"].fillna(0).sum()
                ),
                "Boundary Screens": int(
                    successful["Boundary Screen Exceeded"].fillna(False).sum()
                ),
                "Runtime Seconds": group["Runtime Seconds"].fillna(0).sum(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["Physical Violation Rate"] = (
        summary["Physical Violations"]
        / summary["Predicted Hours"].replace(0, np.nan)
    )
    # Rank only complete paired results. Raw physical excursions remain a
    # separate adoption/QC metric because production applies physical bounds.
    summary["Complete"] = summary["Successful Cases"].eq(
        summary["Planned Cases"]
    )
    return summary


def winner_table(summary: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in summary.groupby(groups):
        eligible = group[group["Complete"]].dropna(
            subset=["Mean Normalized RMSE"]
        )
        if eligible.empty:
            continue
        winner = eligible.sort_values(
            ["Mean Normalized RMSE", "Mean Normalized MAE", "Method"]
        ).iloc[0]
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({
            **dict(zip(groups, keys)),
            "Winner": winner.Method,
            "Winner Mean Normalized RMSE": winner["Mean Normalized RMSE"],
            "Winner Mean Normalized MAE": winner["Mean Normalized MAE"],
        })
    return pd.DataFrame(rows)


def stability_table(
    seed_winners: pd.DataFrame,
    season_winners: pd.DataFrame,
    station_winners: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for parameter in NON_PPT_MET_PARAMS:
        for gap_class in GAP_CLASSES:
            seed_group = seed_winners[
                seed_winners["Parameter"].eq(parameter)
                & seed_winners["Gap Class"].eq(gap_class)
            ]
            season_group = season_winners[
                season_winners["Parameter"].eq(parameter)
                & season_winners["Gap Class"].eq(gap_class)
            ]
            station_group = station_winners[
                station_winners["Parameter"].eq(parameter)
                & station_winners["Gap Class"].eq(gap_class)
            ]
            seed_counts = seed_group["Winner"].value_counts()
            season_counts = season_group["Winner"].value_counts()
            station_counts = station_group["Winner"].value_counts()
            seed_method = seed_counts.index[0] if len(seed_counts) else ""
            season_method = season_counts.index[0] if len(season_counts) else ""
            station_method = station_counts.index[0] if len(station_counts) else ""
            seed_wins = int(seed_counts.iloc[0]) if len(seed_counts) else 0
            season_wins = int(season_counts.iloc[0]) if len(season_counts) else 0
            station_wins = int(station_counts.iloc[0]) if len(station_counts) else 0
            stable = (
                seed_method == season_method == station_method
                and seed_wins >= 3
                and season_wins >= 3
                and station_wins >= 4
                and seed_group["Seed"].nunique() == 4
                and season_group["Season"].nunique() == 4
                and station_group["Station"].nunique() == len(SOURCE_STATIONS)
            )
            current = ROBUSTNESS_BASELINE_METHODS[parameter][gap_class]
            if stable and seed_method == current:
                decision = "current_method_confirmed"
            elif stable:
                decision = "candidate_change_requires_targeted_confirmation"
            else:
                decision = "unstable_retain_current"
            rows.append({
                "Parameter": parameter,
                "Gap Class": gap_class,
                "Current Production Method": current,
                "Seed-Mode Winner": seed_method,
                "Seed Wins": seed_wins,
                "Season-Mode Winner": season_method,
                "Season Wins": season_wins,
                "Station-Mode Winner": station_method,
                "Station Wins": station_wins,
                "Stable Across Seeds, Seasons, and Stations": stable,
                "Robust Winner": seed_method if stable else "",
                "Decision": decision,
                "Final Production Method": current,
                "Production Script Changed": False,
            })
    return pd.DataFrame(rows)


def sarimax_matched_summary(
    sarimax: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for parameter, group in sarimax.groupby("Parameter"):
        ok = group[group["Status"].eq("ok")]
        case_ids = set(ok["Case ID"])
        matched = detail[
            detail["Case ID"].isin(case_ids)
            & detail["Status"].eq("ok")
        ]
        method_means = matched.groupby("Method")["Normalized RMSE"].mean()
        rows.append({
            "Parameter": parameter,
            "Planned SARIMAX Cases": len(group),
            "Successful SARIMAX Cases": len(ok),
            "SARIMAX Mean Normalized RMSE": ok["Normalized RMSE"].mean(),
            "Best Standard Method": (
                method_means.idxmin() if len(method_means) else ""
            ),
            "Best Standard Mean Normalized RMSE": (
                method_means.min() if len(method_means) else np.nan
            ),
            "SARIMAX Physical Violations": int(
                ok["Physical Violations"].fillna(0).sum()
            ),
        })
    return pd.DataFrame(rows)


def sample_targeted_cases(
    frames: dict[str, pd.DataFrame],
    seed: int = TARGETED_SEED,
) -> pd.DataFrame:
    """Sample one independent very-long Wind speed gap per station and season."""
    low, high = GAP_LENGTH_RANGES[TARGETED_GAP_CLASS]
    rows = []
    for station in SOURCE_STATIONS:
        series = frames[station][TARGETED_PARAMETER]
        runs = [run for run in observed_runs(series) if run[1] - run[0] + 1 >= low + 2]
        if not runs:
            raise RuntimeError(f"No eligible very-long run for {station}")
        occupied = np.zeros(len(series), dtype=bool)
        rng = np.random.default_rng(
            stable_seed(TARGETED_VERSION, seed, station, TARGETED_PARAMETER)
        )
        for season in TARGETED_SEASONS:
            selected = None
            for _ in range(50_000):
                run_start, run_end = runs[int(rng.integers(0, len(runs)))]
                max_length = min(high, run_end - run_start - 1)
                if max_length < low:
                    continue
                length = int(rng.integers(low, max_length + 1))
                start = int(rng.integers(run_start + 1, run_end - length + 1))
                end = start + length - 1
                if occupied[max(0, start - 1):min(len(series), end + 2)].any():
                    continue
                index = pd.DatetimeIndex(series.index[start:end + 1])
                if season_for(index) != season:
                    continue
                occupied[start:end + 1] = True
                selected = index
                break
            if selected is None:
                raise RuntimeError(
                    f"Could not sample {season} targeted case for {station}"
                )
            rows.append({
                "Benchmark Version": TARGETED_VERSION,
                "Seed": seed,
                "Case ID": (
                    f"{seed}|{station}|{TARGETED_PARAMETER}|"
                    f"{TARGETED_GAP_CLASS}|{season}|{selected[0].isoformat()}"
                ),
                "Station": station,
                "Parameter": TARGETED_PARAMETER,
                "Gap Class": TARGETED_GAP_CLASS,
                "Sample": 1,
                "Start": selected[0],
                "End": selected[-1],
                "Hours": len(selected),
                "Season": season,
            })
    return pd.DataFrame(rows).sort_values(
        ["Station", "Season", "Start"]
    ).reset_index(drop=True)


def deployment_gap_inventory(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """List actual internal very-long Wind speed gaps that production must fill."""
    rows = []
    for station in SOURCE_STATIONS:
        source = frames[station][TARGETED_PARAMETER]
        missing = source.isna().to_numpy()
        changes = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        for start, end in zip(starts, ends):
            index = pd.DatetimeIndex(source.index[start:end + 1])
            if (
                len(index) >= GAP_LENGTH_RANGES[TARGETED_GAP_CLASS][0]
                and start > 0
                and end < len(source) - 1
                and pd.notna(source.iloc[start - 1])
                and pd.notna(source.iloc[end + 1])
            ):
                rows.append({
                    "Station": station,
                    "Parameter": TARGETED_PARAMETER,
                    "Gap Class": TARGETED_GAP_CLASS,
                    "Start": index[0],
                    "End": index[-1],
                    "Hours": len(index),
                })
    return pd.DataFrame(rows)


def deployment_coverage_inventory(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare every actual internal MET gap with the robustness sample range."""
    rows = []
    for station in SOURCE_STATIONS:
        for parameter in NON_PPT_MET_PARAMS:
            source = frames[station][parameter]
            missing = source.isna().to_numpy()
            changes = np.diff(np.r_[False, missing, False].astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1) - 1
            for start, end in zip(starts, ends):
                if start == 0 or end == len(source) - 1:
                    continue
                if pd.isna(source.iloc[start - 1]) or pd.isna(source.iloc[end + 1]):
                    continue
                hours = int(end - start + 1)
                if hours < 24:
                    gap_class_name = "short"
                elif hours < 168:
                    gap_class_name = "medium"
                elif hours < 720:
                    gap_class_name = "long"
                else:
                    gap_class_name = "verylong"
                low, high = GAP_LENGTH_RANGES[gap_class_name]
                if hours < low:
                    coverage = "shorter_than_benchmark_sampling_minimum"
                    action = "no_longer_horizon_extrapolation"
                elif hours > high:
                    coverage = "longer_than_benchmark_maximum"
                    action = "deployment_length_test_required"
                else:
                    coverage = "within_benchmark_range"
                    action = "none"
                rows.append({
                    "Station": station,
                    "Parameter": parameter,
                    "Gap Class": gap_class_name,
                    "Start": source.index[start],
                    "End": source.index[end],
                    "Hours": hours,
                    "Benchmark Minimum Hours": low,
                    "Benchmark Maximum Hours": high,
                    "Coverage Status": coverage,
                    "Action": action,
                    "Current Production Method": (
                        MET_GAP_METHODS[parameter][gap_class_name]
                    ),
                })
    return pd.DataFrame(rows).sort_values(
        ["Parameter", "Station", "Start"]
    ).reset_index(drop=True)


def sample_deployment_length_cases(
    frames: dict[str, pd.DataFrame],
    hours: int,
    seed: int = DEPLOYMENT_SEED,
    parameter: str = TARGETED_PARAMETER,
    benchmark_version: str = DEPLOYMENT_VERSION,
    require_all_stations: bool = True,
) -> pd.DataFrame:
    """Hide one observed segment per station at the requested deployment length."""
    rows = []
    for station in SOURCE_STATIONS:
        series = frames[station][parameter]
        runs = [
            run for run in observed_runs(series)
            if run[1] - run[0] + 1 >= hours + 2
        ]
        if not runs:
            if require_all_stations:
                raise RuntimeError(
                    f"No {hours}-hour observed {parameter} run for {station}"
                )
            continue
        rng = np.random.default_rng(
            stable_seed(benchmark_version, seed, station, hours)
        )
        run_start, run_end = runs[int(rng.integers(0, len(runs)))]
        start = int(rng.integers(run_start + 1, run_end - hours + 1))
        end = start + hours - 1
        index = pd.DatetimeIndex(series.index[start:end + 1])
        rows.append({
            "Benchmark Version": benchmark_version,
            "Seed": seed,
            "Case ID": (
                f"{seed}|{station}|{parameter}|deployment|"
                f"{hours}|{index[0].isoformat()}"
            ),
            "Station": station,
            "Parameter": parameter,
            "Gap Class": TARGETED_GAP_CLASS,
            "Sample": 1,
            "Start": index[0],
            "End": index[-1],
            "Hours": len(index),
            "Season": season_for(index),
        })
    return pd.DataFrame(rows).sort_values("Station").reset_index(drop=True)


def targeted_confirmation_summaries(
    cases: pd.DataFrame,
    detail: pd.DataFrame,
    deployment_gaps: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build paired summaries and apply adoption gates fixed before the run."""
    overall = aggregate_model_scores(cases, detail, ["Parameter"])
    seasonal = aggregate_model_scores(cases, detail, ["Season"])
    station = aggregate_model_scores(cases, detail, ["Station"])
    season_winners = winner_table(seasonal, ["Season"])
    station_winners = winner_table(station, ["Station"])

    successful = detail[detail["Status"].eq("ok")]
    paired_scores = successful.pivot(
        index="Case ID", columns="Method", values="Normalized RMSE"
    ).reset_index()
    paired = cases.merge(paired_scores, on="Case ID", how="left")
    paired["Winner"] = np.where(
        paired["xgboost"] < paired["donor_regression"],
        "xgboost",
        np.where(
            paired["donor_regression"] < paired["xgboost"],
            "donor_regression",
            "tie",
        ),
    )
    paired["Candidate Relative Improvement"] = (
        paired["donor_regression"] - paired["xgboost"]
    ) / paired["donor_regression"].replace(0, np.nan)

    current = overall[overall["Method"].eq("donor_regression")].iloc[0]
    candidate = overall[overall["Method"].eq("xgboost")].iloc[0]
    relative_improvement = float(
        (current["Mean Normalized RMSE"] - candidate["Mean Normalized RMSE"])
        / current["Mean Normalized RMSE"]
    )
    candidate_season_wins = int(season_winners["Winner"].eq("xgboost").sum())
    candidate_station_wins = int(station_winners["Winner"].eq("xgboost").sum())
    candidate_case_wins = int(paired["Winner"].eq("xgboost").sum())
    complete_gate = bool(current["Complete"] and candidate["Complete"])
    error_gate = bool(
        candidate["Mean Normalized RMSE"] < current["Mean Normalized RMSE"]
        and relative_improvement >= TARGETED_MIN_RELATIVE_IMPROVEMENT
    )
    season_gate = candidate_season_wins >= 3
    station_gate = candidate_station_wins >= 4
    paired_gate = candidate_case_wins > len(cases) / 2
    physical_gate = bool(
        candidate["Physical Violation Rate"]
        <= current["Physical Violation Rate"]
        and candidate["Physical Violation Rate"]
        <= TARGETED_MAX_PHYSICAL_VIOLATION_RATE
    )
    boundary_gate = bool(
        candidate["Boundary Screens"] == 0
        and candidate["Boundary Screens"] <= current["Boundary Screens"]
    )
    statistical_passed = all((
        complete_gate,
        error_gate,
        season_gate,
        station_gate,
        paired_gate,
        physical_gate,
        boundary_gate,
    ))
    maximum_test_hours = int(cases["Hours"].max())
    maximum_deployment_hours = int(deployment_gaps["Hours"].max())
    deployment_length_gate = maximum_test_hours >= maximum_deployment_hours
    passed = statistical_passed and deployment_length_gate
    decision = pd.DataFrame([{
        "Parameter": TARGETED_PARAMETER,
        "Gap Class": TARGETED_GAP_CLASS,
        "Independent Seed": TARGETED_SEED,
        "Current Method": "donor_regression",
        "Candidate Method": "xgboost",
        "Planned Paired Cases": len(cases),
        "All Fits Complete": complete_gate,
        "Current Mean Normalized RMSE": current["Mean Normalized RMSE"],
        "Candidate Mean Normalized RMSE": candidate["Mean Normalized RMSE"],
        "Candidate Relative NRMSE Improvement": relative_improvement,
        "Minimum Required Relative Improvement": TARGETED_MIN_RELATIVE_IMPROVEMENT,
        "Error Gate Passed": error_gate,
        "Candidate Season Wins": candidate_season_wins,
        "Season Gate Passed": season_gate,
        "Candidate Station Wins": candidate_station_wins,
        "Station Gate Passed": station_gate,
        "Candidate Paired-Case Wins": candidate_case_wins,
        "Paired-Case Gate Passed": paired_gate,
        "Current Physical Violation Rate": current["Physical Violation Rate"],
        "Candidate Physical Violation Rate": candidate["Physical Violation Rate"],
        "Maximum Candidate Physical Violation Rate": TARGETED_MAX_PHYSICAL_VIOLATION_RATE,
        "Physical Gate Passed": physical_gate,
        "Current Boundary Screens": int(current["Boundary Screens"]),
        "Candidate Boundary Screens": int(candidate["Boundary Screens"]),
        "Boundary Gate Passed": boundary_gate,
        "Statistical Gates Passed": statistical_passed,
        "Maximum Evaluated Gap Hours": maximum_test_hours,
        "Maximum Deployment Gap Hours": maximum_deployment_hours,
        "Deployment Length Gate Passed": deployment_length_gate,
        "All Adoption Gates Passed": passed,
        "Decision": (
            "candidate_confirmed_for_production"
            if passed
            else "range_test_passed_requires_deployment_length_confirmation"
            if statistical_passed
            else "retain_current_after_targeted_confirmation"
        ),
        "Recommended Production Method": "xgboost" if passed else "donor_regression",
        "Production Script Changed": (
            MET_GAP_METHODS[TARGETED_PARAMETER][TARGETED_GAP_CLASS]
            == "xgboost"
        ),
    }])
    return {
        "overall_summary": overall,
        "season_summary": seasonal,
        "season_winners": season_winners,
        "station_summary": station,
        "station_winners": station_winners,
        "paired_cases": paired,
        "decision": decision,
    }


def deployment_confirmation_summaries(
    cases: pd.DataFrame,
    detail: pd.DataFrame,
    current_method: str = "donor_regression",
    candidate_method: str = "xgboost",
    seed: int = DEPLOYMENT_SEED,
) -> dict[str, pd.DataFrame]:
    """Score the exact-length deployment test and produce the final decision."""
    overall = aggregate_model_scores(cases, detail, ["Parameter"])
    station = aggregate_model_scores(cases, detail, ["Station"])
    station_winners = winner_table(station, ["Station"])
    successful = detail[detail["Status"].eq("ok")]
    scores = successful.pivot(
        index="Case ID", columns="Method", values="Normalized RMSE"
    ).reset_index()
    paired = cases.merge(scores, on="Case ID", how="left")
    paired["Winner"] = np.where(
        paired[candidate_method] < paired[current_method],
        candidate_method,
        np.where(
            paired[current_method] < paired[candidate_method],
            current_method,
            "tie",
        ),
    )
    paired["Candidate Relative Improvement"] = (
        paired[current_method] - paired[candidate_method]
    ) / paired[current_method].replace(0, np.nan)

    current = overall[overall["Method"].eq(current_method)].iloc[0]
    candidate = overall[overall["Method"].eq(candidate_method)].iloc[0]
    relative_improvement = float(
        (current["Mean Normalized RMSE"] - candidate["Mean Normalized RMSE"])
        / current["Mean Normalized RMSE"]
    )
    candidate_station_wins = int(
        station_winners["Winner"].eq(candidate_method).sum()
    )
    candidate_case_wins = int(paired["Winner"].eq(candidate_method).sum())
    complete_gate = bool(current["Complete"] and candidate["Complete"])
    error_gate = bool(
        candidate["Mean Normalized RMSE"] < current["Mean Normalized RMSE"]
        and relative_improvement >= TARGETED_MIN_RELATIVE_IMPROVEMENT
    )
    station_gate = candidate_station_wins >= 4
    paired_gate = candidate_case_wins > len(cases) / 2
    physical_gate = bool(
        candidate["Physical Violation Rate"]
        <= current["Physical Violation Rate"]
        and candidate["Physical Violation Rate"]
        <= TARGETED_MAX_PHYSICAL_VIOLATION_RATE
    )
    boundary_gate = bool(
        candidate["Boundary Screens"] == 0
        and candidate["Boundary Screens"] <= current["Boundary Screens"]
    )
    passed = all((
        complete_gate,
        error_gate,
        station_gate,
        paired_gate,
        physical_gate,
        boundary_gate,
    ))
    decision = pd.DataFrame([{
        "Parameter": cases["Parameter"].iloc[0],
        "Gap Class": TARGETED_GAP_CLASS,
        "Independent Seed": seed,
        "Current Method": current_method,
        "Candidate Method": candidate_method,
        "Deployment-Matched Gap Hours": int(cases["Hours"].iloc[0]),
        "Planned Paired Cases": len(cases),
        "All Fits Complete": complete_gate,
        "Current Mean Normalized RMSE": current["Mean Normalized RMSE"],
        "Candidate Mean Normalized RMSE": candidate["Mean Normalized RMSE"],
        "Candidate Relative NRMSE Improvement": relative_improvement,
        "Minimum Required Relative Improvement": TARGETED_MIN_RELATIVE_IMPROVEMENT,
        "Error Gate Passed": error_gate,
        "Candidate Station Wins": candidate_station_wins,
        "Station Gate Passed": station_gate,
        "Candidate Paired-Case Wins": candidate_case_wins,
        "Paired-Case Gate Passed": paired_gate,
        "Current Physical Violation Rate": current["Physical Violation Rate"],
        "Candidate Physical Violation Rate": candidate["Physical Violation Rate"],
        "Maximum Candidate Physical Violation Rate": TARGETED_MAX_PHYSICAL_VIOLATION_RATE,
        "Physical Gate Passed": physical_gate,
        "Current Boundary Screens": int(current["Boundary Screens"]),
        "Candidate Boundary Screens": int(candidate["Boundary Screens"]),
        "Boundary Gate Passed": boundary_gate,
        "All Adoption Gates Passed": passed,
        "Decision": (
            "candidate_confirmed_for_production"
            if passed else "retain_current_after_deployment_length_confirmation"
        ),
        "Recommended Production Method": (
            candidate_method if passed else current_method
        ),
        "Production Script Changed": (
            MET_GAP_METHODS[cases["Parameter"].iloc[0]][TARGETED_GAP_CLASS]
            == candidate_method
            and passed
        ),
    }])
    return {
        "overall_summary": overall,
        "station_summary": station,
        "station_winners": station_winners,
        "paired_cases": paired,
        "decision": decision,
    }


def deployment_confirmation_figure(
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
    filename_stem: str = "met_wind_verylong_deployment_confirmation",
) -> tuple[Path, Path]:
    """Create the exact deployment-length paired comparison figure."""
    decision = summaries["decision"].iloc[0]
    current_method = decision["Current Method"]
    candidate_method = decision["Candidate Method"]
    parameter = decision["Parameter"]
    station = summaries["station_summary"].pivot(
        index="Station", columns="Method", values="Mean Normalized RMSE"
    ).reindex(SOURCE_STATIONS)
    overall = summaries["overall_summary"].set_index("Method")
    station.loc["Overall"] = {
        current_method: overall.loc[current_method, "Mean Normalized RMSE"],
        candidate_method: overall.loc[candidate_method, "Mean Normalized RMSE"],
    }
    positions = np.arange(len(station))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    axis.bar(
        positions - width / 2,
        station[current_method],
        width,
        label=f"Current: {METHOD_LABELS[current_method]}",
        color=METHOD_COLORS[current_method],
    )
    axis.bar(
        positions + width / 2,
        station[candidate_method],
        width,
        label=f"Candidate: {METHOD_LABELS[candidate_method]}",
        color=METHOD_COLORS[candidate_method],
    )
    axis.set_xticks(positions, station.index)
    axis.set_ylabel("Normalized RMSE")
    axis.set_title(
        f"Deployment-Length Confirmation: Very-Long {parameter}",
        fontsize=14,
        fontweight="bold",
    )
    axis.legend(frameon=False, loc="upper center", ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    fig.text(
        0.5,
        0.02,
        (
            f"Seed {int(decision['Independent Seed'])}; one paired "
            f"{int(decision['Deployment-Matched Gap Hours']):,}-hour gap per station; "
            f"relative NRMSE improvement: "
            f"{decision['Candidate Relative NRMSE Improvement']:.1%}; "
            f"decision: {decision['Decision']}."
        ),
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.10, right=0.98)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{filename_stem}.png"
    pdf = output_dir / f"{filename_stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def targeted_confirmation_figure(
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Create paired station and season panels for the targeted decision."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.1))
    panels = (
        (axes[0], summaries["station_summary"], "Station", "A. Station-level confirmation"),
        (axes[1], summaries["season_summary"], "Season", "B. Seasonal confirmation"),
    )
    for axis, table, group_column, title in panels:
        pivot = table.pivot(
            index=group_column, columns="Method", values="Mean Normalized RMSE"
        )
        order = list(SOURCE_STATIONS) if group_column == "Station" else list(TARGETED_SEASONS)
        pivot = pivot.reindex(order)
        positions = np.arange(len(pivot))
        width = 0.36
        axis.bar(
            positions - width / 2,
            pivot["donor_regression"],
            width,
            label="Current: donor regression",
            color=METHOD_COLORS["donor_regression"],
        )
        axis.bar(
            positions + width / 2,
            pivot["xgboost"],
            width,
            label="Candidate: XGBoost",
            color=METHOD_COLORS["xgboost"],
        )
        axis.set_xticks(positions, pivot.index)
        axis.set_ylabel("Mean normalized RMSE")
        axis.set_title(title, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    decision = summaries["decision"].iloc[0]
    fig.suptitle(
        "Independent Confirmation: Very-Long Wind Speed Gaps",
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.02,
        (
            f"Independent seed {TARGETED_SEED}; {len(SOURCE_STATIONS)} stations; "
            f"one paired 720-1440 h gap per season and station. "
            f"Relative NRMSE improvement: "
            f"{decision['Candidate Relative NRMSE Improvement']:.1%}. "
            f"Decision: {decision['Decision']}."
        ),
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.76, bottom=0.15, left=0.08, right=0.98, wspace=0.27)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "met_wind_verylong_targeted_confirmation.png"
    pdf = output_dir / "met_wind_verylong_targeted_confirmation.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run_targeted_confirmation(
    seed: int = TARGETED_SEED,
    base_dir: Path | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Run the independent current-versus-candidate Wind speed check."""
    base_dir = base_dir or pipeline_dir()
    output_dir = (
        base_dir / "model_comparison_reports" / "met_robustness"
        / "targeted_confirmation"
    )
    frames, _ = load_qc_masked_frames(base_dir)
    deployment_gaps = deployment_gap_inventory(frames)
    if deployment_gaps.empty:
        raise RuntimeError("No actual very-long Wind speed deployment gap found")
    deployment_hours = int(deployment_gaps["Hours"].max())
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "met_wind_verylong_deployment_gap_inventory.csv"
    deployment_gaps.to_csv(inventory_path, index=False)

    case_path = output_dir / "met_wind_verylong_targeted_cases.csv"
    if case_path.exists():
        cached = pd.read_csv(case_path, parse_dates=["Start", "End"])
        if (
            cached["Benchmark Version"].eq(TARGETED_VERSION).all()
            and cached["Seed"].eq(seed).all()
        ):
            cases = cached
        else:
            cases = sample_targeted_cases(frames, seed)
    else:
        cases = sample_targeted_cases(frames, seed)
    cases.to_csv(case_path, index=False)
    detail_path = output_dir / "met_wind_verylong_targeted_detail.csv"
    detail = run_seed(
        frames,
        cases,
        seed,
        detail_path,
        selected_methods=TARGETED_METHODS,
        benchmark_version=TARGETED_VERSION,
    )
    range_summaries = targeted_confirmation_summaries(
        cases, detail, deployment_gaps
    )
    paths = {
        "deployment_gap_inventory": inventory_path,
        "range_cases": case_path,
        "range_detail": detail_path,
    }
    for name, table in range_summaries.items():
        path = output_dir / f"met_wind_verylong_targeted_{name}.csv"
        table.to_csv(path, index=False)
        paths[f"range_{name}"] = path
    range_png, range_pdf = targeted_confirmation_figure(
        range_summaries, output_dir
    )

    deployment_case_path = (
        output_dir / "met_wind_verylong_deployment_cases.csv"
    )
    if deployment_case_path.exists():
        cached = pd.read_csv(
            deployment_case_path, parse_dates=["Start", "End"]
        )
        if (
            cached["Benchmark Version"].eq(DEPLOYMENT_VERSION).all()
            and cached["Seed"].eq(DEPLOYMENT_SEED).all()
            and cached["Hours"].eq(deployment_hours).all()
        ):
            deployment_cases = cached
        else:
            deployment_cases = sample_deployment_length_cases(
                frames, deployment_hours
            )
    else:
        deployment_cases = sample_deployment_length_cases(
            frames, deployment_hours
        )
    deployment_cases.to_csv(deployment_case_path, index=False)
    deployment_detail_path = (
        output_dir / "met_wind_verylong_deployment_detail.csv"
    )
    deployment_detail = run_seed(
        frames,
        deployment_cases,
        DEPLOYMENT_SEED,
        deployment_detail_path,
        selected_methods=TARGETED_METHODS,
        benchmark_version=DEPLOYMENT_VERSION,
    )
    deployment_summaries = deployment_confirmation_summaries(
        deployment_cases, deployment_detail
    )
    paths["deployment_cases"] = deployment_case_path
    paths["deployment_detail"] = deployment_detail_path
    for name, table in deployment_summaries.items():
        path = output_dir / f"met_wind_verylong_deployment_{name}.csv"
        table.to_csv(path, index=False)
        paths[f"deployment_{name}"] = path
    deployment_png, deployment_pdf = deployment_confirmation_figure(
        deployment_summaries, output_dir
    )
    return {
        "range_cases": cases,
        "range_detail": detail,
        "range_decision": range_summaries["decision"],
        "deployment_gap_inventory": deployment_gaps,
        "deployment_cases": deployment_cases,
        "deployment_detail": deployment_detail,
        "deployment_decision": deployment_summaries["decision"],
        "decision": deployment_summaries["decision"],
        "paths": paths,
        "range_figure_png": range_png,
        "range_figure_pdf": range_pdf,
        "figure_png": deployment_png,
        "figure_pdf": deployment_pdf,
    }


def best_complete_alternative(
    detail: pd.DataFrame,
    parameter: str,
    current_method: str,
) -> tuple[str, float]:
    """Select the lowest-NRMSE complete robustness method other than current."""
    subset = detail[
        detail["Parameter"].eq(parameter)
        & detail["Gap Class"].eq(TARGETED_GAP_CLASS)
    ]
    planned = subset["Case ID"].nunique()
    rows = []
    for method, group in subset.groupby("Method"):
        successful = group[group["Status"].eq("ok")]
        if method == current_method or len(successful) != planned:
            continue
        rows.append((method, float(successful["Normalized RMSE"].mean())))
    if not rows:
        raise RuntimeError(f"No complete alternative method for {parameter}")
    return min(rows, key=lambda item: (item[1], item[0]))


def deployment_coverage_figure(
    inventory: pd.DataFrame,
    decisions: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Summarize actual gap coverage and exact-length model decisions."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2))
    status_order = (
        "within_benchmark_range",
        "shorter_than_benchmark_sampling_minimum",
        "longer_than_benchmark_maximum",
    )
    status_labels = {
        "within_benchmark_range": "Within range",
        "shorter_than_benchmark_sampling_minimum": "Shorter than sample minimum",
        "longer_than_benchmark_maximum": "Longer than sample maximum",
    }
    status_colors = {
        "within_benchmark_range": "#4C78A8",
        "shorter_than_benchmark_sampling_minimum": "#A9A9A9",
        "longer_than_benchmark_maximum": "#D9822B",
    }
    counts = (
        inventory.groupby(["Gap Class", "Coverage Status"]).size()
        .unstack(fill_value=0)
        .reindex(GAP_CLASSES, fill_value=0)
    )
    positions = np.arange(len(GAP_CLASSES))
    width = 0.24
    for offset, status in enumerate(status_order):
        values = counts[status] if status in counts else pd.Series(0, index=counts.index)
        bars = axes[0].bar(
            positions + (offset - 1) * width,
            values,
            width,
            label=status_labels[status],
            color=status_colors[status],
        )
        for bar, value in zip(bars, values):
            if value:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.12,
                    str(int(value)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions, ["Short", "Medium", "Long", "Very long"])
    axes[0].set_ylabel("Actual internal gap segments (log scale)")
    axes[0].set_title("A. Deployment coverage", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.7)

    decision_plot = decisions.copy()
    parameter_order = {
        parameter: order for order, parameter in enumerate(NON_PPT_MET_PARAMS)
    }
    decision_plot["_parameter_order"] = decision_plot["Parameter"].map(
        parameter_order
    )
    decision_plot = decision_plot.sort_values(
        ["_parameter_order", "Deployment-Matched Gap Hours"]
    )
    repeated = decision_plot["Parameter"].map(
        decision_plot["Parameter"].value_counts()
    ).gt(1)
    decision_plot["Plot Label"] = decision_plot["Parameter"]
    decision_plot.loc[repeated, "Plot Label"] = (
        decision_plot.loc[repeated, "Parameter"]
        + " ("
        + decision_plot.loc[repeated, "Deployment-Matched Gap Hours"]
        .map(lambda value: f"{int(value):,}")
        + " h)"
    )
    improvement = decision_plot.set_index("Plot Label")[
        "Candidate Relative NRMSE Improvement"
    ] * 100
    colors = ["#D9822B" if value > 0 else "#777777" for value in improvement]
    bars = axes[1].barh(np.arange(len(improvement)), improvement, color=colors)
    axes[1].axvline(
        TARGETED_MIN_RELATIVE_IMPROVEMENT * 100,
        color="#B22222",
        linestyle="--",
        linewidth=1.4,
        label="Required improvement (5%)",
    )
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(improvement)), improvement.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Candidate relative NRMSE improvement (%)")
    axes[1].set_title("B. Exact deployment-length tests", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="x", color="#DDDDDD", linewidth=0.7)
    for bar, value in zip(bars, improvement):
        # Keep large negative labels inside the bar so they do not collide
        # with the parameter labels at the left edge of the panel.
        label_x = value + (2 if value >= 0 else max(4, abs(value) * 0.03))
        axes[1].text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            ha="left",
            va="center",
            fontsize=8,
        )
    fig.suptitle(
        "TxSON MET Deployment-Gap Coverage Audit",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        (
            f"{len(inventory):,} internal gaps audited; "
            f"{int(inventory['Coverage Status'].eq('longer_than_benchmark_maximum').sum())} "
            "segments exceeded the benchmark maximum. Exact-length tests retained "
            "all current production methods."
        ),
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.08, right=0.98, wspace=0.32)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "met_deployment_coverage_publication.png"
    pdf = output_dir / "met_deployment_coverage_publication.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run_deployment_coverage_audit(
    base_dir: Path | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Audit all actual MET gaps and test every over-range very-long class."""
    base_dir = base_dir or pipeline_dir()
    report_root = base_dir / "model_comparison_reports" / "met_robustness"
    output_dir = report_root / "deployment_coverage"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, _ = load_qc_masked_frames(base_dir)
    inventory = deployment_coverage_inventory(frames)
    summary = (
        inventory.groupby(
            ["Parameter", "Gap Class", "Coverage Status", "Current Production Method"],
            as_index=False,
        )
        .agg(
            Segments=("Hours", "size"),
            Total_Hours=("Hours", "sum"),
            Minimum_Hours=("Hours", "min"),
            Maximum_Hours=("Hours", "max"),
        )
        .rename(columns={
            "Total_Hours": "Total Hours",
            "Minimum_Hours": "Minimum Hours",
            "Maximum_Hours": "Maximum Hours",
        })
    )
    inventory_path = output_dir / "met_deployment_gap_coverage_inventory.csv"
    summary_path = output_dir / "met_deployment_gap_coverage_summary.csv"
    inventory.to_csv(inventory_path, index=False)
    summary.to_csv(summary_path, index=False)

    robustness_detail = pd.read_csv(
        report_root / "met_robustness_detail.csv"
    )
    robustness_stability = pd.read_csv(
        report_root / "met_robustness_stability.csv"
    )
    over_range = inventory[
        inventory["Coverage Status"].eq("longer_than_benchmark_maximum")
    ]
    decisions = []
    test_paths: dict[str, Path] = {}
    for parameter in NON_PPT_MET_PARAMS:
        parameter_gaps = over_range[over_range["Parameter"].eq(parameter)]
        if parameter_gaps.empty:
            continue
        current_method = MET_GAP_METHODS[parameter][TARGETED_GAP_CLASS]
        candidate_method, candidate_robustness_nrmse = best_complete_alternative(
            robustness_detail, parameter, current_method
        )
        actual_gap_hours = sorted(parameter_gaps["Hours"].astype(int).unique())
        maximum_hours = max(actual_gap_hours)
        slug = parameter.lower().replace(" ", "_")
        base_row = robustness_stability[
            robustness_stability["Parameter"].eq(parameter)
            & robustness_stability["Gap Class"].eq(TARGETED_GAP_CLASS)
        ].iloc[0]

        for test_hours in actual_gap_hours:
            if parameter == TARGETED_PARAMETER and test_hours == 12_368:
                existing_dir = report_root / "targeted_confirmation"
                decision = pd.read_csv(
                    existing_dir / "met_wind_verylong_deployment_decision.csv"
                )
                decision["Confirmation Source"] = (
                    "existing_independent_seed_127"
                )
                decision["Eligible Test Stations"] = int(
                    decision["Planned Paired Cases"].iloc[0]
                )
            else:
                version = f"{COVERAGE_VERSION}-{slug}-{test_hours}"
                file_stem = (
                    slug
                    if len(actual_gap_hours) == 1 or test_hours == maximum_hours
                    else f"{slug}_{test_hours}h"
                )
                case_path = output_dir / f"{file_stem}_deployment_cases.csv"
                if case_path.exists():
                    cached = pd.read_csv(
                        case_path, parse_dates=["Start", "End"]
                    )
                    if (
                        cached["Benchmark Version"].eq(version).all()
                        and cached["Seed"].eq(COVERAGE_SEED).all()
                        and cached["Hours"].eq(test_hours).all()
                    ):
                        cases = cached
                    else:
                        cases = sample_deployment_length_cases(
                            frames,
                            test_hours,
                            seed=COVERAGE_SEED,
                            parameter=parameter,
                            benchmark_version=version,
                            require_all_stations=False,
                        )
                else:
                    cases = sample_deployment_length_cases(
                        frames,
                        test_hours,
                        seed=COVERAGE_SEED,
                        parameter=parameter,
                        benchmark_version=version,
                        require_all_stations=False,
                    )
                if len(cases) < 4:
                    raise RuntimeError(
                        f"Only {len(cases)} stations support the {parameter} "
                        f"{test_hours}-hour deployment-length test"
                    )
                cases.to_csv(case_path, index=False)
                detail_path = output_dir / f"{file_stem}_deployment_detail.csv"
                detail = run_seed(
                    frames,
                    cases,
                    COVERAGE_SEED,
                    detail_path,
                    selected_methods=(current_method, candidate_method),
                    benchmark_version=version,
                )
                test_summaries = deployment_confirmation_summaries(
                    cases,
                    detail,
                    current_method=current_method,
                    candidate_method=candidate_method,
                    seed=COVERAGE_SEED,
                )
                for name, table in test_summaries.items():
                    path = output_dir / f"{file_stem}_deployment_{name}.csv"
                    table.to_csv(path, index=False)
                    test_paths[f"{file_stem}_{name}"] = path
                figure_png, figure_pdf = deployment_confirmation_figure(
                    test_summaries,
                    output_dir,
                    filename_stem=f"{file_stem}_deployment_confirmation",
                )
                test_paths[f"{file_stem}_figure_png"] = figure_png
                test_paths[f"{file_stem}_figure_pdf"] = figure_pdf
                decision = test_summaries["decision"].copy()
                decision["Confirmation Source"] = (
                    "deployment_coverage_seed_128"
                )
                decision["Eligible Test Stations"] = len(cases)

            decision["Actual Over-Range Segments For Parameter"] = len(
                parameter_gaps
            )
            decision["Actual Segments At Tested Length"] = int(
                parameter_gaps["Hours"].eq(test_hours).sum()
            )
            decision["Maximum Actual Gap Hours"] = maximum_hours
            decision["Robustness Decision"] = base_row["Decision"]
            decision["Alternative Robustness Mean NRMSE"] = (
                candidate_robustness_nrmse
            )
            decision["Coverage Recommendation"] = np.where(
                decision["All Adoption Gates Passed"],
                "candidate_requires_length_specific_method_review",
                "retain_current_method",
            )
            decisions.append(decision)

    decision_table = pd.concat(decisions, ignore_index=True)
    decision_path = output_dir / "met_deployment_length_decisions.csv"
    decision_table.to_csv(decision_path, index=False)
    figure_png, figure_pdf = deployment_coverage_figure(
        inventory, decision_table, output_dir
    )
    return {
        "inventory": inventory,
        "summary": summary,
        "decisions": decision_table,
        "inventory_path": inventory_path,
        "summary_path": summary_path,
        "decision_path": decision_path,
        "figure_png": figure_png,
        "figure_pdf": figure_pdf,
        "test_paths": test_paths,
    }


def publication_figure(
    stability: pd.DataFrame,
    output_dir: Path,
) -> tuple[plt.Figure, Path, Path]:
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.9), gridspec_kw={"width_ratios": [1.65, 0.75]}
    )
    matrix = stability.copy()
    parameter_order = list(NON_PPT_MET_PARAMS)
    gap_order = list(GAP_CLASSES)
    for row_index, parameter in enumerate(parameter_order):
        for column_index, gap_class in enumerate(gap_order):
            row = matrix[
                matrix["Parameter"].eq(parameter)
                & matrix["Gap Class"].eq(gap_class)
            ].iloc[0]
            method = row["Robust Winner"] or row["Seed-Mode Winner"]
            color = METHOD_COLORS.get(method, "#D9D9D9")
            alpha = (
                1.0
                if row["Stable Across Seeds, Seasons, and Stations"]
                else 0.35
            )
            rectangle = plt.Rectangle(
                (column_index, row_index), 1, 1,
                facecolor=color, alpha=alpha,
                edgecolor=(
                    "#1B5E20"
                    if row["Decision"] == "current_method_confirmed"
                    else "#B36B00"
                    if row["Decision"] == "candidate_change_requires_targeted_confirmation"
                    else "#666666"
                ),
                linewidth=2.5,
            )
            axes[0].add_patch(rectangle)
            axes[0].text(
                column_index + 0.5,
                row_index + 0.42,
                METHOD_SHORT_LABELS.get(method, "No winner"),
                ha="center", va="center", fontsize=9, fontweight="bold",
            )
            axes[0].text(
                column_index + 0.5,
                row_index + 0.70,
                f"S {row['Seed Wins']}/4   Y {row['Season Wins']}/4   N {row['Station Wins']}/6",
                ha="center", va="center", fontsize=7.5,
            )
    axes[0].set_xlim(0, len(gap_order))
    axes[0].set_ylim(len(parameter_order), 0)
    axes[0].set_xticks(
        np.arange(len(gap_order)) + 0.5,
        ["Short\n6-23 h", "Medium\n24-167 h", "Long\n168-719 h", "Very long\n720-1440 h"],
    )
    axes[0].set_yticks(np.arange(len(parameter_order)) + 0.5, parameter_order)
    axes[0].tick_params(length=0)
    axes[0].set_title("A. Winner stability by parameter and gap class", fontweight="bold")
    axes[0].spines[:].set_visible(False)

    counts = stability["Decision"].value_counts()
    categories = [
        ("Confirmed", "current_method_confirmed", "#2E7D32"),
        ("Needs\nconfirmation", "candidate_change_requires_targeted_confirmation", "#D9822B"),
        ("Unstable", "unstable_retain_current", "#777777"),
    ]
    labels = [item[0] for item in categories]
    values = [int(counts.get(item[1], 0)) for item in categories]
    colors = [item[2] for item in categories]
    bars = axes[1].bar(np.arange(3), values, color=colors, width=0.62)
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylim(0, max(values + [1]) * 1.25)
    axes[1].set_ylabel("Parameter-gap combinations")
    axes[1].set_title("B. Robustness decision", fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.7)
    for bar, value in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            str(value),
            ha="center", fontweight="bold",
        )

    method_legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color,
               markeredgecolor=color, markersize=8, label=METHOD_LABELS[method])
        for method, color in METHOD_COLORS.items()
    ]
    fig.legend(
        handles=method_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=6,
        frameon=False,
    )
    fig.suptitle(
        "TxSON Non-Ppt MET Robustness Across Seeds, Seasons, and Stations",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.text(
        0.09, 0.055,
        "Cell counts: S = seeds won (of 4), Y = seasons won (of 4), "
        "N = stations won (of 6). Faded cells do not pass all stability gates.",
        fontsize=8.5,
    )
    fig.text(
        0.09, 0.025,
        "Green border = current method confirmed; orange border = stable "
        "alternative requiring independent confirmation.",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.76, bottom=0.20, left=0.09, right=0.98, wspace=0.28)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "met_model_robustness_publication.png"
    pdf = output_dir / "met_model_robustness_publication.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return fig, png, pdf


def write_summaries(
    cases: pd.DataFrame,
    detail: pd.DataFrame,
    sarimax: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame | Path]:
    seed_summary = aggregate_model_scores(
        cases, detail, ["Seed", "Parameter", "Gap Class"]
    )
    seed_winners = winner_table(
        seed_summary, ["Seed", "Parameter", "Gap Class"]
    )
    seasonal_cases = cases.copy()
    season_summary = aggregate_model_scores(
        seasonal_cases, detail, ["Parameter", "Gap Class", "Season"]
    )
    season_winners = winner_table(
        season_summary, ["Parameter", "Gap Class", "Season"]
    )
    station_summary = aggregate_model_scores(
        cases, detail, ["Station", "Parameter", "Gap Class"]
    )
    station_winners = winner_table(
        station_summary, ["Station", "Parameter", "Gap Class"]
    )
    stability = stability_table(
        seed_winners, season_winners, station_winners
    )
    sarimax_summary = sarimax_matched_summary(sarimax, detail)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "cases": cases,
        "detail": detail,
        "seed_summary": seed_summary,
        "seed_winners": seed_winners,
        "season_summary": season_summary,
        "season_winners": season_winners,
        "station_summary": station_summary,
        "station_winners": station_winners,
        "stability": stability,
        "final_method_map": stability[[
            "Parameter", "Gap Class", "Current Production Method",
            "Robust Winner", "Decision", "Final Production Method",
            "Production Script Changed",
        ]],
        "sarimax_detail": sarimax,
        "sarimax_summary": sarimax_summary,
        "qc_exclusions": audit,
    }
    paths = {}
    for name, table in tables.items():
        path = output_dir / f"met_robustness_{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    figure, png, pdf = publication_figure(stability, output_dir)
    plt.close(figure)
    return {**tables, "paths": paths, "figure_png": png, "figure_pdf": pdf}


def run_benchmark(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    run_sarimax: bool = True,
    base_dir: Path | None = None,
) -> dict[str, pd.DataFrame | Path]:
    base_dir = base_dir or pipeline_dir()
    output_dir = base_dir / "model_comparison_reports" / "met_robustness"
    frames, audit = load_qc_masked_frames(base_dir)
    case_frames = []
    detail_frames = []
    for seed in seeds:
        case_path = output_dir / "seeds" / f"met_seed_{seed}_cases.csv"
        if case_path.exists():
            cached = pd.read_csv(case_path, parse_dates=["Start", "End"])
            if cached["Benchmark Version"].eq(BENCHMARK_VERSION).all():
                cases = cached
            else:
                cases = sample_seed_cases(frames, seed)
        else:
            cases = sample_seed_cases(frames, seed)
        case_path.parent.mkdir(parents=True, exist_ok=True)
        cases.to_csv(case_path, index=False)
        detail_path = output_dir / "seeds" / f"met_seed_{seed}_detail.csv"
        detail = run_seed(frames, cases, seed, detail_path)
        case_frames.append(cases)
        detail_frames.append(detail)
    all_cases = pd.concat(case_frames, ignore_index=True)
    all_detail = pd.concat(detail_frames, ignore_index=True)
    sarimax_path = output_dir / "met_robustness_sarimax_detail.csv"
    if run_sarimax:
        sarimax = run_sarimax_subset(frames, all_cases, sarimax_path)
    elif sarimax_path.exists():
        sarimax = pd.read_csv(sarimax_path, parse_dates=["Start", "End"])
    else:
        sarimax = pd.DataFrame()
    return write_summaries(
        all_cases, all_detail, sarimax, audit, output_dir
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run four-seed robustness testing for non-Ppt MET models."
    )
    parser.add_argument("--seed", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--skip-sarimax", action="store_true",
        help="Skip the separate matched SARIMAX screening subset.",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Validate QC-aware artificial-gap sampling without fitting models.",
    )
    parser.add_argument(
        "--targeted-confirmation", action="store_true",
        help=(
            "Run the independent seed-126 donor-versus-XGBoost confirmation "
            "for very-long Wind speed gaps."
        ),
    )
    parser.add_argument(
        "--deployment-coverage", action="store_true",
        help=(
            "Audit actual non-Ppt MET gap lengths and run exact-length tests "
            "for every actual gap length beyond the robustness range."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = pipeline_dir()
    if args.deployment_coverage:
        reports = run_deployment_coverage_audit(base_dir=base_dir)
        print(reports["summary"].to_string(index=False))
        print("\nDeployment-length decisions")
        print(reports["decisions"][[
            "Parameter",
            "Current Method",
            "Candidate Method",
            "Deployment-Matched Gap Hours",
            "Candidate Relative NRMSE Improvement",
            "Candidate Station Wins",
            "Eligible Test Stations",
            "All Adoption Gates Passed",
            "Coverage Recommendation",
        ]].to_string(index=False))
        print(f"Coverage reports: {reports['decision_path'].parent}")
        return
    if args.targeted_confirmation:
        reports = run_targeted_confirmation(base_dir=base_dir)
        decision = reports["decision"].iloc[0]
        print(decision.to_string())
        print(f"Targeted figure: {reports['figure_png']}")
        return
    frames, audit = load_qc_masked_frames(base_dir)
    if args.sample_only:
        cases = pd.concat(
            [sample_seed_cases(frames, seed) for seed in args.seed],
            ignore_index=True,
        )
        print(
            f"Sampled {len(cases)} QC-eligible gaps across {len(args.seed)} seeds; "
            f"excluded {int(audit['Observed Hours Excluded'].sum())} observed QC hours."
        )
        print(cases.groupby(["Seed", "Gap Class"]).size().to_string())
        return
    reports = run_benchmark(
        seeds=args.seed,
        run_sarimax=not args.skip_sarimax,
        base_dir=base_dir,
    )
    stability = reports["stability"]
    print(stability["Decision"].value_counts().to_string())
    print(f"Publication figure: {reports['figure_png']}")


if __name__ == "__main__":
    main()

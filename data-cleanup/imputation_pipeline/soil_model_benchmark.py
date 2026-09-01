"""Artificial-gap benchmark for TxSON soil moisture and temperature."""
from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


SOIL_MOISTURE = ["SWC_5", "SWC_10", "SWC_20", "SWC_50"]
SOIL_TEMPERATURE = ["T_5", "T_10", "T_20", "T_50"]
SOIL_PARAMETERS = SOIL_MOISTURE + SOIL_TEMPERATURE
BENCHMARK_VERSION = "2026-08-07-soil-v2"
TARGETED_CONFIRMATION_VERSION = "2026-08-07-soil-targeted-v1"
PRODUCTION_ADAPTER_VERSION = "2026-08-07-soil-production-adapter-v2"
PRODUCTION_SARIMAX_TIMEOUT_SECONDS = 180
GAP_CLASSES = ["short", "medium", "long", "verylong"]
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
GAP_LENGTH_RANGES = {
    "short": (6, 23),
    "medium": (24, 167),
    "long": (168, 719),
    "verylong": (720, 1440),
}
STANDARD_METHODS = [
    "interpolation",
    "climatology",
    "donor_regression",
    "random_forest",
    "xgboost",
]
METHOD_LABELS = {
    "interpolation": "PCHIP / time interpolation",
    "climatology": "Monthly-hour climatology",
    "donor_regression": "Donor regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "sarimax": "SARIMAX",
}
GAP_LABELS = {
    "short": "Short\n6-23 h",
    "medium": "Medium\n24-167 h",
    "long": "Long\n168-719 h",
    "verylong": "Very long\n720-1440 h",
}
PHYSICAL_BOUNDS = {
    "soil_moisture": (0.0, 0.6),
    "soil_temperature": (-30.0, 60.0),
}
METHOD_COLORS = {
    "interpolation": "#4C78A8",
    "climatology": "#8C8C8C",
    "donor_regression": "#59A14F",
    "random_forest": "#F28E2B",
    "xgboost": "#E15759",
    "sarimax": "#B279A2",
}
SHORT_METHOD_LABELS = {
    "interpolation": "Interpolation",
    "climatology": "Climatology",
    "donor_regression": "Donor regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "sarimax": "SARIMAX",
}
CURRENT_SOIL_METHODS = {
    "short": "interpolation",
    "medium": "sarimax",
    "long": "xgboost",
    "verylong": "donor_regression",
}


def stable_rng(*parts: object) -> np.random.Generator:
    token = "|".join(map(str, parts)).encode()
    seed = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
    return np.random.default_rng(seed)


def family_for(parameter: str) -> str:
    return "soil_moisture" if parameter.startswith("SWC_") else "soil_temperature"


def discover_stations(cleaned_dir: Path) -> list[str]:
    stations = []
    for path in cleaned_dir.glob("Station*_cleaned_data.csv"):
        station = path.name.removeprefix("Station").removesuffix("_cleaned_data.csv")
        if station and not station.isdigit():
            stations.append(station)
    return sorted(stations)


def read_station(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, low_memory=False)
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[frame.index.notna()]
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


def load_soil_frames(
    cleaned_dir: Path,
    stations: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    selected = list(stations) if stations is not None else discover_stations(cleaned_dir)
    frames = {}
    for station in selected:
        path = cleaned_dir / f"Station{station}_cleaned_data.csv"
        if path.exists():
            frames[station] = read_station(path)
    if not frames:
        raise FileNotFoundError(f"No cleaned station files found under {cleaned_dir}")
    return frames


def load_benchmark_exclusions(base_dir: Path) -> pd.DataFrame:
    sensor_path = base_dir / "sensor_qc_reports" / "sensor_qc_decisions.csv"
    manual_path = base_dir / "manual_qc_masks.csv"
    if not sensor_path.exists() or not manual_path.exists():
        raise FileNotFoundError(
            "Soil benchmark requires sensor_qc_reports/sensor_qc_decisions.csv "
            "and manual_qc_masks.csv from the completed soil QC workflow."
        )

    sensor = pd.read_csv(sensor_path)
    sensor = sensor[sensor["QC Decision"].eq("bad_sensor_candidate")].copy()
    sensor["Source"] = "sensor_qc_decisions"
    sensor["Reason"] = sensor["Decision Reason"]

    manual = pd.read_csv(manual_path)
    manual = manual[manual["Decision"].eq("mask_and_refill")].copy()
    manual["Source"] = "manual_qc_masks"
    manual["Reason"] = manual["Reason"].fillna(manual.get("Notes"))

    columns = ["Station", "Parameter", "Start", "End", "Source", "Reason"]
    exclusions = pd.concat([sensor[columns], manual[columns]], ignore_index=True)
    exclusions["Station"] = exclusions["Station"].astype(str)
    exclusions["Start"] = pd.to_datetime(exclusions["Start"], errors="coerce")
    exclusions["End"] = pd.to_datetime(exclusions["End"], errors="coerce")
    return exclusions.sort_values(["Station", "Parameter", "Start"]).reset_index(drop=True)


def apply_benchmark_exclusions(
    frames: dict[str, pd.DataFrame],
    exclusions: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    audited = []
    for _, row in exclusions.iterrows():
        station, parameter = row["Station"], row["Parameter"]
        removed = 0
        if station in frames and parameter in frames[station]:
            mask = frames[station].index.to_series().between(row["Start"], row["End"])
            mask &= frames[station][parameter].notna()
            removed = int(mask.sum())
            frames[station].loc[mask, parameter] = np.nan
        audited.append({**row.to_dict(), "Observed Hours Removed": removed})
    return frames, pd.DataFrame(audited)


def observed_runs(series: pd.Series) -> list[tuple[int, int]]:
    good = series.notna().to_numpy()
    if not good.any():
        return []
    edges = np.flatnonzero(np.r_[True, good[1:] != good[:-1], True])
    return [
        (int(start), int(end))
        for start, end in zip(edges[:-1], edges[1:])
        if good[start]
    ]


def sample_artificial_gaps(
    frames: dict[str, pd.DataFrame],
    parameters: Iterable[str] = SOIL_PARAMETERS,
    gap_classes: Iterable[str] = GAP_CLASSES,
    gaps_per_combination: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for station, frame in sorted(frames.items()):
        for parameter in parameters:
            if parameter not in frame or frame[parameter].notna().sum() == 0:
                continue
            series = frame[parameter]
            runs = observed_runs(series)
            for gap_class in gap_classes:
                low, high = GAP_LENGTH_RANGES[gap_class]
                rng = stable_rng(random_seed, station, parameter, gap_class)
                for sample in range(1, gaps_per_combination + 1):
                    max_feasible = max((end - start - 2 for start, end in runs), default=0)
                    feasible_high = min(high, max_feasible)
                    if feasible_high < low:
                        continue
                    length = int(rng.integers(low, feasible_high + 1))
                    eligible = [run for run in runs if run[1] - run[0] >= length + 2]
                    run_start, run_end = eligible[int(rng.integers(0, len(eligible)))]
                    first = int(rng.integers(run_start + 1, run_end - length))
                    last = first + length - 1
                    gap_index = series.index[first : last + 1]
                    rows.append(
                        {
                            "Case ID": f"{station}|{parameter}|{gap_class}|{sample}",
                            "Station": station,
                            "Parameter": parameter,
                            "Family": family_for(parameter),
                            "Gap Class": gap_class,
                            "Sample": sample,
                            "Start": gap_index[0],
                            "End": gap_index[-1],
                            "Hours": len(gap_index),
                        }
                    )
    return pd.DataFrame(rows)


def donor_matrix(frames: dict[str, pd.DataFrame], parameter: str) -> pd.DataFrame:
    columns = {
        station: frame[parameter]
        for station, frame in frames.items()
        if parameter in frame and frame[parameter].notna().any()
    }
    return pd.concat(columns, axis=1).sort_index()


def select_donor(
    target: pd.Series,
    donors: pd.DataFrame,
    station: str,
    gap_index: pd.DatetimeIndex,
    rng: np.random.Generator,
    max_rows: int = 10_000,
) -> tuple[pd.Series | None, float]:
    candidates = donors.drop(columns=[station], errors="ignore").reindex(target.index)
    training = target.copy()
    training.loc[gap_index] = np.nan
    index = training.dropna().index
    if len(index) > max_rows:
        index = pd.DatetimeIndex(rng.choice(index.to_numpy(), max_rows, replace=False)).sort_values()
    sampled = candidates.reindex(index)
    counts = sampled.notna().sum()
    correlations = sampled.corrwith(training.reindex(index)).where(counts >= 168)
    if correlations.dropna().empty:
        return None, np.nan
    name = correlations.abs().idxmax()
    return candidates[name], float(correlations[name])


def donor_prediction(
    masked: pd.Series,
    donors: pd.DataFrame,
    station: str,
    gap_index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series | None, float]:
    donor, correlation = select_donor(masked, donors, station, gap_index, rng)
    if donor is None:
        return pd.Series(np.nan, index=gap_index), None, correlation
    overlap = masked.notna() & donor.notna()
    train_index = masked.index[overlap]
    if len(train_index) > 20_000:
        train_index = pd.DatetimeIndex(
            rng.choice(train_index.to_numpy(), 20_000, replace=False)
        )
    if len(train_index) < 168:
        return pd.Series(np.nan, index=gap_index), donor, correlation
    model = LinearRegression().fit(
        donor.reindex(train_index).to_numpy().reshape(-1, 1),
        masked.reindex(train_index).to_numpy(),
    )
    donor_gap = donor.reindex(gap_index)
    network_gap = donors.drop(columns=[station], errors="ignore").reindex(gap_index).median(axis=1)
    donor_gap = donor_gap.fillna(network_gap)
    prediction = pd.Series(np.nan, index=gap_index, dtype=float)
    valid = donor_gap.notna()
    if valid.any():
        prediction.loc[valid] = model.predict(
            donor_gap.loc[valid].to_numpy().reshape(-1, 1)
        )
    return prediction, donor, correlation


def base_features(
    frame: pd.DataFrame,
    parameter: str,
    donors: pd.DataFrame,
    station: str,
) -> pd.DataFrame:
    index = frame.index
    hour = index.hour.to_numpy()
    day = index.dayofyear.to_numpy()
    features = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_sin": np.sin(2 * np.pi * day / 365.25),
            "day_cos": np.cos(2 * np.pi * day / 365.25),
            "trend": np.arange(len(index), dtype=float) / max(len(index), 1),
        },
        index=index,
    )
    covariates = [
        col for col in SOIL_PARAMETERS + ["Ppt", "Tair", "Srad"]
        if col != parameter and col in frame
    ]
    for column in covariates:
        features[f"same_station_{column}"] = frame[column]

    network = donors.drop(columns=[station], errors="ignore").reindex(index)
    features["network_median"] = network.median(axis=1)
    features["network_mean"] = network.mean(axis=1)
    features["network_count"] = network.notna().sum(axis=1)
    return features.replace([np.inf, -np.inf], np.nan)


def model_features(
    base: pd.DataFrame,
    masked: pd.Series,
    donor: pd.Series | None,
) -> pd.DataFrame:
    features = base.copy()
    if donor is not None:
        features["selected_donor"] = donor.reindex(features.index)
    for lag in (1, 24, 168):
        features[f"target_lag_{lag}"] = masked.shift(lag)
    features["target_mean_24"] = masked.shift(1).rolling(24, min_periods=6).mean()
    features["target_mean_168"] = masked.shift(1).rolling(168, min_periods=24).mean()
    return features


def interpolation_prediction(
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    parameter: str,
) -> pd.Series:
    before = masked.loc[: gap_index[0]].dropna().tail(4)
    after = masked.loc[gap_index[-1] :].dropna().head(4)
    anchors = pd.concat([before, after])
    anchors = anchors.loc[~anchors.index.duplicated()].sort_index()
    if len(anchors) < 2 or anchors.index[0] >= gap_index[0] or anchors.index[-1] <= gap_index[-1]:
        return pd.Series(np.nan, index=gap_index)
    origin = anchors.index[0]
    x = (anchors.index - origin) / pd.Timedelta(hours=1)
    x_new = (gap_index - origin) / pd.Timedelta(hours=1)
    if parameter.startswith("SWC_") and len(anchors) >= 4:
        values = PchipInterpolator(x, anchors.to_numpy(), extrapolate=False)(x_new)
    else:
        values = np.interp(x_new, x, anchors.to_numpy())
    return pd.Series(values, index=gap_index, dtype=float)


def climatology_prediction(masked: pd.Series, gap_index: pd.DatetimeIndex) -> pd.Series:
    observed = masked.dropna()
    table = pd.DataFrame(
        {
            "value": observed.to_numpy(),
            "month": observed.index.month,
            "hour": observed.index.hour,
        }
    )
    monthly_hour = table.groupby(["month", "hour"])["value"].mean()
    hourly = table.groupby("hour")["value"].mean()
    values = []
    for timestamp in gap_index:
        value = monthly_hour.get((timestamp.month, timestamp.hour), np.nan)
        if pd.isna(value):
            value = hourly.get(timestamp.hour, observed.mean())
        values.append(value)
    return pd.Series(values, index=gap_index, dtype=float)


def training_index(
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    rng: np.random.Generator,
    max_rows: int,
) -> pd.DatetimeIndex:
    available = masked.dropna().index
    local_start = gap_index[0] - pd.Timedelta(days=90)
    local_end = gap_index[-1] + pd.Timedelta(days=90)
    local = available[(available >= local_start) & (available <= local_end)]
    local_limit = min(len(local), max_rows // 2)
    if len(local) > local_limit:
        local = pd.DatetimeIndex(rng.choice(local.to_numpy(), local_limit, replace=False))
    remaining = available.difference(local)
    global_limit = max_rows - len(local)
    if len(remaining) > global_limit:
        remaining = pd.DatetimeIndex(
            rng.choice(remaining.to_numpy(), global_limit, replace=False)
        )
    return local.append(remaining).unique().sort_values()


def tree_prediction(
    method: str,
    features: pd.DataFrame,
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    rng: np.random.Generator,
    max_train_rows: int,
    n_estimators: int,
) -> pd.Series:
    train_index = training_index(masked, gap_index, rng, max_train_rows)
    if len(train_index) < 168:
        return pd.Series(np.nan, index=gap_index)
    if method == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
    elif method == "xgboost" and XGBRegressor is not None:
        estimator = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=4,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
    else:
        return pd.Series(np.nan, index=gap_index)
    train_features = features.reindex(train_index)
    usable_columns = train_features.columns[train_features.notna().any()]
    if len(usable_columns) == 0:
        return pd.Series(np.nan, index=gap_index)
    model = make_pipeline(SimpleImputer(strategy="median"), estimator)
    model.fit(train_features[usable_columns], masked.reindex(train_index))
    return pd.Series(
        model.predict(features.reindex(gap_index)[usable_columns]),
        index=gap_index,
    )


def sarimax_prediction(
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    context_days: int = 7,
    maxiter: int = 50,
) -> tuple[pd.Series, bool, str]:
    train_end = gap_index[0] - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=context_days) + pd.Timedelta(hours=1)
    train = masked.loc[train_start:train_end].reindex(
        pd.date_range(train_start, train_end, freq="h")
    )
    if train.notna().sum() < 24:
        return pd.Series(np.nan, index=gap_index), False, "fewer_than_24_training_hours"
    train = train.interpolate(method="time", limit_direction="both").ffill().bfill()
    train.index.freq = "h"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = SARIMAX(
                train,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 0, 24),
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True,
            )
            fitted = model.fit(method="powell", maxiter=maxiter, disp=False)
        converged = bool(fitted.mle_retvals.get("converged", True))
        converged &= not any(isinstance(item.message, ConvergenceWarning) for item in caught)
        prediction = pd.Series(
            np.asarray(fitted.forecast(steps=len(gap_index)), dtype=float),
            index=gap_index,
        )
        left = masked.get(gap_index[0] - pd.Timedelta(hours=1), np.nan)
        right = masked.get(gap_index[-1] + pd.Timedelta(hours=1), np.nan)
        if pd.notna(left) and pd.notna(right):
            start_delta = left - prediction.iloc[0]
            end_delta = right - prediction.iloc[-1]
            prediction += np.linspace(start_delta, end_delta, len(prediction))
        elif pd.notna(left):
            prediction.iloc[0] = 0.5 * (left + prediction.iloc[0])
        elif pd.notna(right):
            prediction.iloc[-1] = 0.5 * (right + prediction.iloc[-1])
        return prediction, converged, "" if converged else "maximum_likelihood_not_converged"
    except Exception as exc:
        return pd.Series(np.nan, index=gap_index), False, str(exc)


def score_prediction(
    case: pd.Series,
    truth: pd.Series,
    prediction: pd.Series,
    target_training: pd.Series,
    method: str,
    runtime: float,
    donor_correlation: float = np.nan,
    converged: bool | None = None,
    error: str = "",
) -> dict:
    family = case["Family"]
    low, high = PHYSICAL_BOUNDS[family]
    raw = pd.to_numeric(prediction.reindex(truth.index), errors="coerce")
    raw_violations = int(((raw < low) | (raw > high)).sum())
    scored = raw.clip(lower=low, upper=high)
    valid = scored.notna() & truth.notna()
    status = "ok" if valid.all() else "failed"
    if method == "sarimax" and converged is False and valid.all():
        status = "nonconverged"
    if status == "failed" and not error:
        error = "incomplete_prediction"
    if valid.any():
        residual = scored.loc[valid] - truth.loc[valid]
        mae = float(residual.abs().mean())
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        bias = float(residual.mean())
        endpoints = residual.iloc[[0, -1]] if len(residual) > 1 else residual
        boundary_mae = float(endpoints.abs().mean())
    else:
        mae = rmse = bias = boundary_mae = np.nan
    scale = float(target_training.quantile(0.75) - target_training.quantile(0.25))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(target_training.std())
    nrmse = rmse / scale if np.isfinite(rmse) and scale > 1e-9 else np.nan
    return {
        "Benchmark Version": BENCHMARK_VERSION,
        **case.to_dict(),
        "Method": method,
        "Method Label": METHOD_LABELS[method],
        "Status": status,
        "Converged": converged,
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE IQR": nrmse,
        "Bias": bias,
        "Boundary MAE": boundary_mae,
        "Raw Physical Violations": raw_violations,
        "Donor Correlation": donor_correlation,
        "Runtime Seconds": runtime,
        "Error": error,
    }


def run_standard_benchmark(
    frames: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    methods: Iterable[str] = STANDARD_METHODS,
    max_train_rows: int = 5_000,
    n_estimators: int = 60,
    random_seed: int = 42,
    checkpoint_path: Path | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    rows = []
    total = len(cases)
    grouped = cases.sort_values(["Parameter", "Station", "Gap Class"]).groupby(
        ["Parameter", "Station"], sort=False
    )
    processed = 0
    for (parameter, station), group in grouped:
        target = frames[station][parameter]
        donors = donor_matrix(frames, parameter)
        base = base_features(frames[station], parameter, donors, station)
        for _, case in group.iterrows():
            gap_index = pd.date_range(case["Start"], case["End"], freq="h")
            truth = target.reindex(gap_index)
            masked = target.copy()
            masked.loc[gap_index] = np.nan
            rng = stable_rng(random_seed, case["Case ID"])
            donor_pred, donor, correlation = donor_prediction(
                masked, donors, station, gap_index, rng
            )
            features = model_features(base, masked, donor)
            for method in methods:
                started = time.perf_counter()
                error = ""
                try:
                    if method == "interpolation":
                        prediction = interpolation_prediction(masked, gap_index, parameter)
                    elif method == "climatology":
                        prediction = climatology_prediction(masked, gap_index)
                    elif method == "donor_regression":
                        prediction = donor_pred
                    elif method in {"random_forest", "xgboost"}:
                        prediction = tree_prediction(
                            method,
                            features,
                            masked,
                            gap_index,
                            rng,
                            max_train_rows,
                            n_estimators,
                        )
                        if method == "xgboost" and XGBRegressor is None:
                            error = "xgboost_not_installed"
                    else:
                        raise ValueError(f"Unknown method: {method}")
                except Exception as exc:
                    prediction = pd.Series(np.nan, index=gap_index)
                    error = str(exc)
                rows.append(
                    score_prediction(
                        case,
                        truth,
                        prediction,
                        masked.dropna(),
                        method,
                        time.perf_counter() - started,
                        donor_correlation=correlation if method == "donor_regression" else np.nan,
                        error=error,
                    )
                )
            processed += 1
            if processed % progress_every == 0 or processed == total:
                print(f"Standard benchmark: {processed}/{total} artificial gaps")
            if checkpoint_path is not None and processed % progress_every == 0:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    result = pd.DataFrame(rows)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(checkpoint_path, index=False)
    return result


def select_sarimax_cases(
    cases: pd.DataFrame,
    cases_per_parameter: int = 2,
    random_seed: int = 42,
) -> pd.DataFrame:
    selected = []
    medium = cases[cases["Gap Class"].eq("medium")]
    for parameter, group in medium.groupby("Parameter"):
        take = min(cases_per_parameter, len(group))
        selected.append(group.sample(take, random_state=random_seed))
    return pd.concat(selected, ignore_index=True) if selected else medium.head(0)


def run_sarimax_subset(
    frames: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    checkpoint_path: Path | None = None,
    context_days: int = 7,
    maxiter: int = 50,
    progress_every: int = 1,
) -> pd.DataFrame:
    rows = []
    for number, (_, case) in enumerate(cases.iterrows(), start=1):
        target = frames[case["Station"]][case["Parameter"]]
        gap_index = pd.date_range(case["Start"], case["End"], freq="h")
        truth = target.reindex(gap_index)
        masked = target.copy()
        masked.loc[gap_index] = np.nan
        started = time.perf_counter()
        prediction, converged, error = sarimax_prediction(
            masked,
            gap_index,
            context_days=context_days,
            maxiter=maxiter,
        )
        rows.append(
            score_prediction(
                case,
                truth,
                prediction,
                masked.dropna(),
                "sarimax",
                time.perf_counter() - started,
                converged=converged,
                error=error,
            )
        )
        if number % progress_every == 0 or number == len(cases):
            print(f"SARIMAX subset: {number}/{len(cases)} artificial gaps")
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    return pd.DataFrame(rows)


def error_matrix(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    successful = results[results["Status"].eq("ok")]
    matrix = successful.pivot_table(
        index="Parameter",
        columns=["Gap Class", "Method Label"],
        values=metric,
        aggfunc="mean",
    )
    gap_order = [gap for gap in GAP_CLASSES if gap in matrix.columns.get_level_values(0)]
    return matrix.reindex(columns=gap_order, level=0)


def selection_table(results: pd.DataFrame) -> pd.DataFrame:
    successful = results[results["Status"].eq("ok")]
    summary = (
        successful.groupby(["Family", "Parameter", "Gap Class", "Method", "Method Label"])
        .agg(
            Cases=("Case ID", "nunique"),
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_NRMSE_IQR=("NRMSE IQR", "mean"),
            Mean_Boundary_MAE=("Boundary MAE", "mean"),
            Scored_Hours=("Hours", "sum"),
            Physical_Violations=("Raw Physical Violations", "sum"),
            Mean_Runtime_Seconds=("Runtime Seconds", "mean"),
        )
        .reset_index()
    )
    summary["Physical_Violation_Rate"] = (
        summary["Physical_Violations"] / summary["Scored_Hours"]
    )
    max_cases = summary.groupby(["Parameter", "Gap Class"])["Cases"].transform("max")
    eligible = summary[summary["Cases"] >= 0.8 * max_cases].copy()
    winners = eligible.loc[
        eligible.groupby(["Parameter", "Gap Class"])["Mean_NRMSE_IQR"].idxmin()
    ].copy()
    production = {
        "short": "interpolation",
        "medium": "sarimax",
        "long": "xgboost",
        "verylong": "donor_regression",
    }
    winners["Current Production Method"] = winners["Gap Class"].map(production)
    compared_methods = successful.groupby(["Parameter", "Gap Class"])["Method"].agg(set)
    winners["Production Method In Main Benchmark"] = winners.apply(
        lambda row: row["Current Production Method"]
        in compared_methods.get((row["Parameter"], row["Gap Class"]), set()),
        axis=1,
    )
    winners["Matches Current Production"] = winners.apply(
        lambda row: row["Method"] == row["Current Production Method"]
        if row["Production Method In Main Benchmark"]
        else pd.NA,
        axis=1,
    )
    return winners.sort_values(["Parameter", "Gap Class"]).reset_index(drop=True)


def write_reports(
    results: pd.DataFrame,
    cases: pd.DataFrame,
    report_dir: Path,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "detail": report_dir / "soil_model_comparison_detail.csv",
        "cases": report_dir / "soil_artificial_gap_cases.csv",
        "mae": report_dir / "soil_model_comparison_mae_matrix.csv",
        "rmse": report_dir / "soil_model_comparison_rmse_matrix.csv",
        "nrmse": report_dir / "soil_model_comparison_nrmse_matrix.csv",
        "selection": report_dir / "soil_model_selection.csv",
        "sarimax_matched": report_dir / "soil_sarimax_matched_summary.csv",
    }
    results.to_csv(paths["detail"], index=False)
    cases.to_csv(paths["cases"], index=False)
    error_matrix(results, "MAE").to_csv(paths["mae"])
    error_matrix(results, "RMSE").to_csv(paths["rmse"])
    error_matrix(results, "NRMSE IQR").to_csv(paths["nrmse"])
    selection_table(results[results["Method"].ne("sarimax")]).to_csv(
        paths["selection"], index=False
    )
    sarimax_ids = set(results.loc[results["Method"].eq("sarimax"), "Case ID"])
    matched = results[
        results["Case ID"].isin(sarimax_ids) & results["Status"].eq("ok")
    ]
    matched_summary = (
        matched.groupby(["Family", "Method", "Method Label"])
        .agg(
            Cases=("Case ID", "nunique"),
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_NRMSE_IQR=("NRMSE IQR", "mean"),
            Scored_Hours=("Hours", "sum"),
            Physical_Violations=("Raw Physical Violations", "sum"),
            Mean_Runtime_Seconds=("Runtime Seconds", "mean"),
        )
        .reset_index()
    )
    matched_summary["Physical_Violation_Rate"] = (
        matched_summary["Physical_Violations"] / matched_summary["Scored_Hours"]
    )
    matched_summary.to_csv(paths["sarimax_matched"], index=False)
    return paths


def _mean_ci(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_columns)["NRMSE IQR"]
        .agg(["mean", "std", "count"])
        .assign(ci95=lambda x: 1.96 * x["std"].fillna(0) / np.sqrt(x["count"]))
        .reset_index()
    )


def publication_figure(
    results: pd.DataFrame,
    sarimax_case_ids: Iterable[str],
    report_dir: Path,
) -> tuple[plt.Figure, Path, Path]:
    successful = results[results["Status"].eq("ok")].copy()
    standard = successful[successful["Method"].isin(STANDARD_METHODS)]
    subset = successful[successful["Case ID"].isin(set(sarimax_case_ids))]
    top = _mean_ci(standard, ["Family", "Gap Class", "Method", "Method Label"])
    bottom = _mean_ci(subset, ["Family", "Method", "Method Label"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    families = ["soil_moisture", "soil_temperature"]
    titles = ["Soil moisture", "Soil temperature"]
    x = np.arange(len(GAP_CLASSES))
    for column, (family, title) in enumerate(zip(families, titles)):
        axis = axes[0, column]
        data = top[top["Family"].eq(family)]
        for method in STANDARD_METHODS:
            method_data = data[data["Method"].eq(method)].set_index("Gap Class")
            means = method_data["mean"].reindex(GAP_CLASSES)
            cis = method_data["ci95"].reindex(GAP_CLASSES)
            axis.errorbar(
                x,
                means,
                yerr=cis,
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(f"{title}: all sampled gaps", fontweight="bold")
        axis.set_xticks(x, [GAP_LABELS[gap] for gap in GAP_CLASSES])
        axis.set_ylabel("NRMSE / observed IQR")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

        subset_axis = axes[1, column]
        subset_data = bottom[bottom["Family"].eq(family)].sort_values("mean")
        positions = np.arange(len(subset_data))
        subset_axis.barh(
            positions,
            subset_data["mean"],
            xerr=subset_data["ci95"],
            color=[METHOD_COLORS[method] for method in subset_data["Method"]],
            capsize=3,
        )
        subset_axis.set_yticks(
            positions,
            [SHORT_METHOD_LABELS[method] for method in subset_data["Method"]],
        )
        subset_axis.invert_yaxis()
        subset_axis.set_xlabel("NRMSE / observed IQR")
        subset_cases = subset.loc[subset["Family"].eq(family), "Case ID"].nunique()
        subset_axis.set_title(
            f"{title}: matched medium-gap subset (n={subset_cases})",
            fontweight="bold",
        )
        subset_axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        subset_axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "TxSON Soil Artificial-Gap Model Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.12, right=0.98, hspace=0.42, wspace=0.34)
    report_dir.mkdir(parents=True, exist_ok=True)
    png = report_dir / "soil_model_comparison_publication.png"
    pdf = report_dir / "soil_model_comparison_publication.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return fig, png, pdf


def tag_robustness_seed(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["Base Case ID"] = tagged.get("Base Case ID", tagged["Case ID"])
    tagged["Case ID"] = str(seed) + "|" + tagged["Base Case ID"].astype(str)
    tagged["Seed"] = seed
    start = pd.to_datetime(tagged["Start"])
    end = pd.to_datetime(tagged["End"])
    month = (start + (end - start) / 2).dt.month
    tagged["Season"] = np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["MAM", "JJA", "SON"],
        default="DJF",
    )
    return tagged


def robustness_tables(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    successful = results[results["Status"].eq("ok")].copy()
    standard = successful[successful["Method"].isin(STANDARD_METHODS)]

    seed_gap = (
        standard.groupby(["Seed", "Family", "Gap Class", "Method", "Method Label"])
        ["NRMSE IQR"].mean().reset_index(name="Seed Mean NRMSE IQR")
    )
    gap_summary = (
        seed_gap.groupby(["Family", "Gap Class", "Method", "Method Label"])
        ["Seed Mean NRMSE IQR"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Mean NRMSE IQR",
            "std": "Seed SD",
            "min": "Seed Min",
            "max": "Seed Max",
            "count": "Seeds",
        })
    )

    seed_season = (
        standard.groupby(["Seed", "Family", "Season", "Method", "Method Label"])
        ["NRMSE IQR"].mean().reset_index(name="Seed Mean NRMSE IQR")
    )
    season_summary = (
        seed_season.groupby(["Family", "Season", "Method", "Method Label"])
        ["Seed Mean NRMSE IQR"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Mean NRMSE IQR",
            "std": "Seed SD",
            "min": "Seed Min",
            "max": "Seed Max",
            "count": "Seeds",
        })
    )

    seed_gap_season = (
        standard.groupby(
            ["Seed", "Family", "Gap Class", "Season", "Method", "Method Label"]
        )["NRMSE IQR"].mean().reset_index(name="Seed Mean NRMSE IQR")
    )
    gap_season_summary = (
        seed_gap_season.groupby(
            ["Family", "Gap Class", "Season", "Method", "Method Label"]
        )["Seed Mean NRMSE IQR"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Mean NRMSE IQR",
            "std": "Seed SD",
            "min": "Seed Min",
            "max": "Seed Max",
            "count": "Seeds",
        })
    )

    method_seed = (
        standard.groupby(["Seed", "Parameter", "Gap Class", "Method", "Method Label"])
        .agg(Cases=("Case ID", "nunique"), Mean_NRMSE_IQR=("NRMSE IQR", "mean"))
        .reset_index()
    )
    max_cases = method_seed.groupby(["Seed", "Parameter", "Gap Class"])["Cases"].transform("max")
    eligible = method_seed[method_seed["Cases"] >= 0.8 * max_cases]
    seed_winners = eligible.loc[
        eligible.groupby(["Seed", "Parameter", "Gap Class"])["Mean_NRMSE_IQR"].idxmin()
    ].copy()
    counts = (
        seed_winners.groupby(["Parameter", "Gap Class", "Method", "Method Label"])
        .agg(
            Winner_Seeds=("Seed", "nunique"),
            Mean_Winner_NRMSE_IQR=("Mean_NRMSE_IQR", "mean"),
        )
        .reset_index()
    )
    total_seeds = results["Seed"].nunique()
    stability = (
        counts.sort_values(
            ["Parameter", "Gap Class", "Winner_Seeds", "Mean_Winner_NRMSE_IQR"],
            ascending=[True, True, False, True],
        )
        .groupby(["Parameter", "Gap Class"], as_index=False)
        .first()
    )
    stability["Total Seeds"] = total_seeds
    stability["Winner Fraction"] = stability["Winner_Seeds"] / total_seeds
    stability["Stable 3 of 4"] = stability["Winner_Seeds"] >= 3

    parameter_season = (
        standard.groupby(
            ["Parameter", "Gap Class", "Season", "Method", "Method Label"]
        )
        .agg(Cases=("Case ID", "nunique"), Mean_NRMSE_IQR=("NRMSE IQR", "mean"))
        .reset_index()
    )
    max_season_cases = parameter_season.groupby(
        ["Parameter", "Gap Class", "Season"]
    )["Cases"].transform("max")
    season_eligible = parameter_season[
        parameter_season["Cases"] >= 0.8 * max_season_cases
    ]
    season_winners = season_eligible.loc[
        season_eligible.groupby(["Parameter", "Gap Class", "Season"])
        ["Mean_NRMSE_IQR"].idxmin()
    ].copy()
    season_counts = (
        season_winners.groupby(["Parameter", "Gap Class", "Method"])
        ["Season"].nunique().reset_index(name="Winner Seasons")
    )
    total_seasons = (
        season_winners.groupby(["Parameter", "Gap Class"])["Season"]
        .nunique().reset_index(name="Total Seasons")
    )
    stability = stability.merge(
        season_counts,
        on=["Parameter", "Gap Class", "Method"],
        how="left",
    ).merge(total_seasons, on=["Parameter", "Gap Class"], how="left")
    stability["Winner Seasons"] = stability["Winner Seasons"].fillna(0).astype(int)
    stability["Season Winner Fraction"] = (
        stability["Winner Seasons"] / stability["Total Seasons"]
    )
    stability["Stable Across Seeds and Seasons"] = (
        stability["Stable 3 of 4"] & (stability["Winner Seasons"] >= 3)
    )

    sarimax_ids = set(successful.loc[successful["Method"].eq("sarimax"), "Case ID"])
    matched = successful[successful["Case ID"].isin(sarimax_ids)]
    matched_seed = (
        matched.groupby(["Seed", "Family", "Method", "Method Label"])
        ["NRMSE IQR"].mean().reset_index(name="Seed Mean NRMSE IQR")
    )
    matched_summary = (
        matched_seed.groupby(["Family", "Method", "Method Label"])
        ["Seed Mean NRMSE IQR"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "Mean NRMSE IQR",
            "std": "Seed SD",
            "min": "Seed Min",
            "max": "Seed Max",
            "count": "Seeds",
        })
    )
    return {
        "gap_seed": seed_gap,
        "gap_summary": gap_summary,
        "season_seed": seed_season,
        "season_summary": season_summary,
        "gap_season_summary": gap_season_summary,
        "seed_winners": seed_winners,
        "season_winners": season_winners,
        "winner_stability": stability,
        "sarimax_matched": matched_summary,
    }


def write_robustness_reports(
    results: pd.DataFrame,
    cases: pd.DataFrame,
    report_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = robustness_tables(results)
    paths = {
        "cases": report_dir / "soil_robustness_cases.csv",
        "gap_summary": report_dir / "soil_robustness_gap_summary.csv",
        "season_summary": report_dir / "soil_robustness_season_summary.csv",
        "gap_season_summary": report_dir / "soil_robustness_gap_season_summary.csv",
        "seed_winners": report_dir / "soil_robustness_seed_winners.csv",
        "season_winners": report_dir / "soil_robustness_parameter_season_winners.csv",
        "winner_stability": report_dir / "soil_robustness_winner_stability.csv",
        "sarimax_matched": report_dir / "soil_robustness_sarimax_matched_summary.csv",
    }
    cases.to_csv(paths["cases"], index=False)
    for name in paths:
        if name != "cases":
            tables[name].to_csv(paths[name], index=False)
    return tables, paths


def robustness_figure(
    results: pd.DataFrame,
    report_dir: Path,
) -> tuple[plt.Figure, Path, Path]:
    successful = results[
        results["Status"].eq("ok") & results["Method"].isin(STANDARD_METHODS)
    ]
    gap_seed = (
        successful.groupby(["Seed", "Family", "Gap Class", "Method"])["NRMSE IQR"]
        .mean().reset_index()
    )
    season_seed = (
        successful.groupby(["Seed", "Family", "Season", "Method"])["NRMSE IQR"]
        .mean().reset_index()
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    families = ["soil_moisture", "soil_temperature"]
    titles = ["Soil moisture", "Soil temperature"]

    for column, (family, title) in enumerate(zip(families, titles)):
        for row, (data, category, order, subtitle) in enumerate([
            (gap_seed, "Gap Class", GAP_CLASSES, "gap-length robustness"),
            (season_seed, "Season", SEASON_ORDER, "seasonal robustness"),
        ]):
            axis = axes[row, column]
            family_data = data[data["Family"].eq(family)]
            x = np.arange(len(order))
            for method in STANDARD_METHODS:
                method_data = family_data[family_data["Method"].eq(method)]
                grouped = method_data.groupby(category)["NRMSE IQR"].agg(["mean", "std", "count"])
                means = grouped["mean"].reindex(order)
                critical = pd.Series(
                    student_t.ppf(0.975, grouped["count"] - 1),
                    index=grouped.index,
                )
                ci = (critical * grouped["std"] / np.sqrt(grouped["count"])).reindex(order)
                axis.errorbar(
                    x,
                    means,
                    yerr=ci,
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            labels = [GAP_LABELS[item] for item in order] if category == "Gap Class" else order
            axis.set_xticks(x, labels)
            axis.set_ylim(bottom=0)
            axis.set_ylabel("NRMSE / observed IQR")
            axis.set_title(f"{title}: {subtitle}", fontweight="bold")
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False)
    seed_count = results["Seed"].nunique()
    fig.suptitle(
        f"TxSON Soil Model Robustness Across {seed_count} Seeds",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.10, right=0.98, hspace=0.42, wspace=0.28)
    report_dir.mkdir(parents=True, exist_ok=True)
    png = report_dir / "soil_model_robustness_publication.png"
    pdf = report_dir / "soil_model_robustness_publication.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return fig, png, pdf


def proposed_method_map(stability: pd.DataFrame) -> pd.DataFrame:
    """Convert robust winners into a conservative, not-yet-applied method map."""
    table = stability.copy()
    stable = table["Stable Across Seeds and Seasons"].eq(True)
    stable |= table["Stable Across Seeds and Seasons"].astype(str).str.lower().eq("true")
    table["Stable Across Seeds and Seasons"] = stable
    table["Current Production Method"] = table["Gap Class"].map(CURRENT_SOIL_METHODS)
    table["Robust Winner"] = table["Method"]
    table["Proposed Method"] = table["Current Production Method"]
    table.loc[stable, "Proposed Method"] = table.loc[stable, "Robust Winner"]
    table["Requires Targeted Confirmation"] = (
        stable & table["Proposed Method"].ne(table["Current Production Method"])
    )
    table["Proposal Status"] = np.select(
        [
            ~stable,
            table["Requires Targeted Confirmation"],
        ],
        [
            "retain_current_unstable_evidence",
            "targeted_confirmation_required",
        ],
        default="retain_current_benchmark_agrees",
    )
    columns = [
        "Parameter",
        "Gap Class",
        "Current Production Method",
        "Robust Winner",
        "Proposed Method",
        "Winner_Seeds",
        "Total Seeds",
        "Winner Seasons",
        "Total Seasons",
        "Stable Across Seeds and Seasons",
        "Requires Targeted Confirmation",
        "Proposal Status",
    ]
    return table[columns].sort_values(["Parameter", "Gap Class"]).reset_index(drop=True)


def select_targeted_confirmation_cases(
    cases: pd.DataFrame,
    proposal: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Select one independent case per season for each proposed method change."""
    changed = proposal[proposal["Requires Targeted Confirmation"]].copy()
    rows = []
    for _, decision in changed.iterrows():
        available = cases[
            cases["Parameter"].eq(decision["Parameter"])
            & cases["Gap Class"].eq(decision["Gap Class"])
        ]
        for season in SEASON_ORDER:
            options = available[available["Season"].eq(season)].sort_values(
                ["Station", "Start"]
            )
            if options.empty:
                continue
            rng = stable_rng(
                TARGETED_CONFIRMATION_VERSION,
                random_seed,
                decision["Parameter"],
                decision["Gap Class"],
                season,
            )
            selected = options.iloc[int(rng.integers(0, len(options)))].to_dict()
            selected.update(
                {
                    "Confirmation Version": TARGETED_CONFIRMATION_VERSION,
                    "Current Production Method": decision["Current Production Method"],
                    "Proposed Method": decision["Proposed Method"],
                }
            )
            rows.append(selected)
    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    return selected.sort_values(
        ["Parameter", "Gap Class", "Season", "Station"]
    ).reset_index(drop=True)


def run_targeted_confirmation(
    frames: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    random_seed: int,
    checkpoint_dir: Path | None = None,
    max_train_rows: int = 5_000,
    n_estimators: int = 60,
) -> pd.DataFrame:
    """Run only the current and proposed method needed for each selected case."""
    parts = []
    for method in STANDARD_METHODS:
        subset = cases[
            cases["Current Production Method"].eq(method)
            | cases["Proposed Method"].eq(method)
        ]
        if subset.empty:
            continue
        checkpoint = (
            checkpoint_dir / f"soil_targeted_{method}.csv"
            if checkpoint_dir is not None
            else None
        )
        parts.append(
            run_standard_benchmark(
                frames,
                subset,
                methods=[method],
                max_train_rows=max_train_rows,
                n_estimators=n_estimators,
                random_seed=random_seed,
                checkpoint_path=checkpoint,
                progress_every=10,
            )
        )

    sarimax_cases = cases[
        cases["Current Production Method"].eq("sarimax")
        | cases["Proposed Method"].eq("sarimax")
    ]
    if not sarimax_cases.empty:
        checkpoint = (
            checkpoint_dir / "soil_targeted_sarimax.csv"
            if checkpoint_dir is not None
            else None
        )
        parts.append(
            run_sarimax_subset(
                frames,
                sarimax_cases,
                checkpoint_path=checkpoint,
                progress_every=1,
            )
        )
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def targeted_confirmation_summary(
    results: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the independent error, season, bounds, and boundary gates."""
    rows = []
    for (parameter, gap_class), planned in cases.groupby(["Parameter", "Gap Class"]):
        current = planned["Current Production Method"].iat[0]
        proposed = planned["Proposed Method"].iat[0]
        case_ids = set(planned["Case ID"])
        relevant = results[results["Case ID"].isin(case_ids)]
        current_rows = relevant[
            relevant["Method"].eq(current) & relevant["Status"].eq("ok")
        ]
        proposed_rows = relevant[
            relevant["Method"].eq(proposed) & relevant["Status"].eq("ok")
        ]
        paired = current_rows[["Case ID", "NRMSE IQR"]].merge(
            proposed_rows[["Case ID", "NRMSE IQR", "Season"]],
            on="Case ID",
            suffixes=(" Current", " Proposed"),
        )
        planned_cases = len(planned)
        planned_seasons = planned["Season"].nunique()
        proposed_wins = int(
            (paired["NRMSE IQR Proposed"] < paired["NRMSE IQR Current"]).sum()
        )
        proposed_season_wins = int(
            paired.loc[
                paired["NRMSE IQR Proposed"] < paired["NRMSE IQR Current"],
                "Season",
            ].nunique()
        )
        current_nrmse = current_rows["NRMSE IQR"].mean()
        proposed_nrmse = proposed_rows["NRMSE IQR"].mean()
        current_boundary = current_rows["Boundary MAE"].mean()
        proposed_boundary = proposed_rows["Boundary MAE"].mean()
        boundary_tolerance = 0.01 if parameter.startswith("SWC_") else 1.0
        complete = (
            planned_cases == 4
            and planned_seasons == 4
            and len(current_rows) == planned_cases
            and len(proposed_rows) == planned_cases
        )
        error_gate = bool(
            np.isfinite(proposed_nrmse)
            and np.isfinite(current_nrmse)
            and proposed_nrmse < current_nrmse
            and proposed_wins >= 3
            and proposed_season_wins >= 3
        )
        physical_gate = int(proposed_rows["Raw Physical Violations"].sum()) == 0
        boundary_gate = bool(
            np.isfinite(proposed_boundary)
            and np.isfinite(current_boundary)
            and proposed_boundary <= current_boundary + boundary_tolerance
        )
        confirmed = complete and error_gate and physical_gate and boundary_gate
        if confirmed:
            status = "confirmed_for_production_trial"
        elif not complete:
            status = "incomplete_retain_current"
        elif not error_gate:
            status = "error_gate_failed_retain_current"
        elif not physical_gate:
            status = "physical_gate_failed_retain_current"
        else:
            status = "boundary_gate_failed_retain_current"
        relative_improvement = (
            (current_nrmse - proposed_nrmse) / current_nrmse
            if np.isfinite(current_nrmse) and current_nrmse > 0
            else np.nan
        )
        rows.append(
            {
                "Parameter": parameter,
                "Gap Class": gap_class,
                "Current Production Method": current,
                "Proposed Method": proposed,
                "Planned Cases": planned_cases,
                "Planned Seasons": planned_seasons,
                "Current Successful Cases": len(current_rows),
                "Proposed Successful Cases": len(proposed_rows),
                "Current Mean NRMSE IQR": current_nrmse,
                "Proposed Mean NRMSE IQR": proposed_nrmse,
                "Relative NRMSE Improvement": relative_improvement,
                "Proposed Case Wins": proposed_wins,
                "Proposed Season Wins": proposed_season_wins,
                "Current Mean Boundary MAE": current_boundary,
                "Proposed Mean Boundary MAE": proposed_boundary,
                "Boundary Tolerance": boundary_tolerance,
                "Proposed Physical Violations": int(
                    proposed_rows["Raw Physical Violations"].sum()
                ),
                "Complete Four-Season Test": complete,
                "Error Gate": error_gate,
                "Physical Gate": physical_gate,
                "Boundary Gate": boundary_gate,
                "Confirmation Status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["Parameter", "Gap Class"]).reset_index(drop=True)


def finalized_method_map(
    proposal: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> pd.DataFrame:
    """Attach independent confirmation while keeping production unchanged."""
    table = proposal.merge(
        confirmation[
            [
                "Parameter",
                "Gap Class",
                "Relative NRMSE Improvement",
                "Proposed Case Wins",
                "Proposed Season Wins",
                "Confirmation Status",
            ]
        ],
        on=["Parameter", "Gap Class"],
        how="left",
    )
    confirmed = table["Confirmation Status"].eq("confirmed_for_production_trial")
    table["Recommended Method After Confirmation"] = table["Current Production Method"]
    table.loc[confirmed, "Recommended Method After Confirmation"] = table.loc[
        confirmed, "Proposed Method"
    ]
    table["Final Recommendation"] = np.select(
        [
            confirmed,
            table["Requires Targeted Confirmation"],
            table["Stable Across Seeds and Seasons"],
        ],
        [
            "candidate_for_production_trial",
            "retain_current_after_independent_check",
            "retain_current_benchmark_agrees",
        ],
        default="retain_current_unstable_evidence",
    )
    return table.sort_values(["Parameter", "Gap Class"]).reset_index(drop=True)


def write_targeted_confirmation_reports(
    proposal: pd.DataFrame,
    cases: pd.DataFrame,
    results: pd.DataFrame,
    report_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    confirmation = targeted_confirmation_summary(results, cases)
    final_map = finalized_method_map(proposal, confirmation)
    paths = {
        "proposal": report_dir / "soil_proposed_method_map.csv",
        "cases": report_dir / "soil_targeted_confirmation_cases.csv",
        "detail": report_dir / "soil_targeted_confirmation_detail.csv",
        "summary": report_dir / "soil_targeted_confirmation_summary.csv",
        "final_map": report_dir / "soil_method_map_after_confirmation.csv",
    }
    proposal.to_csv(paths["proposal"], index=False)
    cases.to_csv(paths["cases"], index=False)
    results.to_csv(paths["detail"], index=False)
    confirmation.to_csv(paths["summary"], index=False)
    final_map.to_csv(paths["final_map"], index=False)
    return confirmation, final_map, paths


def targeted_confirmation_figure(
    confirmation: pd.DataFrame,
    report_dir: Path,
) -> tuple[plt.Figure, Path, Path]:
    """Draw a publication-ready paired NRMSE comparison for proposed changes."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.5))
    for axis, (is_swc, title) in zip(
        axes,
        [(True, "Soil moisture"), (False, "Soil temperature")],
    ):
        subset = confirmation[
            confirmation["Parameter"].str.startswith("SWC_").eq(is_swc)
        ].copy()
        subset["Gap Order"] = subset["Gap Class"].map(
            {name: number for number, name in enumerate(GAP_CLASSES)}
        )
        subset["Parameter Order"] = subset["Parameter"].map(
            {name: number for number, name in enumerate(SOIL_PARAMETERS)}
        )
        subset = subset.sort_values(["Parameter Order", "Gap Order"])
        y = np.arange(len(subset))
        passed = subset["Confirmation Status"].eq("confirmed_for_production_trial")
        colors = np.where(passed, "#2E7D32", "#C62828")
        for position, (_, row) in enumerate(subset.iterrows()):
            axis.plot(
                [row["Current Mean NRMSE IQR"], row["Proposed Mean NRMSE IQR"]],
                [position, position],
                color=colors[position],
                linewidth=2,
                alpha=0.75,
            )
        axis.scatter(
            subset["Current Mean NRMSE IQR"],
            y,
            marker="s",
            s=42,
            color="#777777",
            label="Current method",
            zorder=3,
        )
        axis.scatter(
            subset["Proposed Mean NRMSE IQR"],
            y,
            marker="o",
            s=50,
            c=colors,
            label="Proposed method",
            zorder=4,
        )
        axis.set_yticks(
            y,
            [
                (
                    f"SWC {parameter.removeprefix('SWC_')} cm | {gap_class.title()}"
                    if parameter.startswith("SWC_")
                    else f"Soil T {parameter.removeprefix('T_')} cm | {gap_class.title()}"
                )
                for parameter, gap_class in zip(
                    subset["Parameter"], subset["Gap Class"]
                )
            ],
        )
        axis.invert_yaxis()
        axis.set_xlabel("Mean NRMSE / observed IQR")
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777",
               markeredgecolor="#777777", label="Current method"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#222222",
               markeredgecolor="#222222", label="Proposed method"),
        Line2D([0], [0], color="#2E7D32", linewidth=2.5, label="Confirmed"),
        Line2D([0], [0], color="#C62828", linewidth=2.5, label="Retain current"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Independent Four-Season Confirmation of Proposed Soil Method Changes",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.82, bottom=0.09, left=0.14, right=0.98, wspace=0.48)
    report_dir.mkdir(parents=True, exist_ok=True)
    png = report_dir / "soil_targeted_confirmation_publication.png"
    pdf = report_dir / "soil_targeted_confirmation_publication.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return fig, png, pdf


def select_production_adapter_smoke_cases(
    targeted_cases: pd.DataFrame,
    final_method_map: pd.DataFrame,
) -> pd.DataFrame:
    """Select the shortest independent case for each confirmed method change."""
    candidates = final_method_map[
        final_method_map["Final Recommendation"].eq("candidate_for_production_trial")
    ][["Parameter", "Gap Class"]]
    selected = (
        targeted_cases.merge(candidates, on=["Parameter", "Gap Class"], how="inner")
        .sort_values(["Parameter", "Gap Class", "Hours", "Case ID"])
        .groupby(["Parameter", "Gap Class"], as_index=False)
        .first()
    )
    selected["Production Adapter Version"] = PRODUCTION_ADAPTER_VERSION
    selected["Smoke Selection"] = "shortest_independent_case"
    return selected.sort_values(["Parameter", "Gap Class"]).reset_index(drop=True)


def _production_interpolation_prediction(
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    parameter: str,
) -> tuple[pd.Series, dict]:
    from Shortgaps import time_interpolate
    from param_config import short_interp_for

    method = short_interp_for(parameter)
    prediction = time_interpolate(
        masked,
        gap_index[0],
        gap_index[-1],
        method=method,
    )
    return prediction.reindex(gap_index), {"Production Detail": method}


def _production_sarimax_prediction(
    frame: pd.DataFrame,
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    parameter: str,
) -> tuple[pd.Series, dict]:
    import signal

    from Mediumgaps import (
        apply_physical_bounds,
        correct_boundary_drift,
        get_exog,
        sarima_forecast,
    )
    from param_config import exog_for

    exog_columns = exog_for(parameter)
    exog = get_exog(frame, prefer=exog_columns)
    def timeout_handler(_signum, _frame):
        raise TimeoutError(
            f"production SARIMAX exceeded {PRODUCTION_SARIMAX_TIMEOUT_SECONDS}s"
        )

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, PRODUCTION_SARIMAX_TIMEOUT_SECONDS)
    try:
        prediction, fitted = sarima_forecast(
            masked,
            gap_index[0],
            gap_index[-1],
            exog,
            ctx_days=7,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    if prediction is None:
        return pd.Series(np.nan, index=gap_index), {
            "Converged": False,
            "Production Detail": "auto_sarimax_failed",
        }
    prediction = correct_boundary_drift(
        prediction,
        masked,
        gap_index[0],
        gap_index[-1],
    )
    prediction = apply_physical_bounds(prediction, parameter)
    converged = bool(fitted.mle_retvals.get("converged", True))
    detail = f"order={fitted.model.order};seasonal={fitted.model.seasonal_order}"
    if exog is not None:
        detail += ";exog=" + "+".join(exog.columns)
    return prediction.reindex(gap_index), {
        "Converged": converged,
        "Production Detail": detail,
    }


def _production_xgboost_prediction(
    frame: pd.DataFrame,
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    parameter: str,
) -> tuple[pd.Series, dict]:
    from Longgaps import (
        apply_physical_bounds,
        correct_boundary_drift,
        ensure_driver_columns,
        rolling_fill,
        train_xgb,
    )

    work = frame.copy()
    work[parameter] = masked
    ensure_driver_columns(work)
    model = train_xgb(work.copy(), parameter)
    prediction = rolling_fill(model, work, gap_index, parameter)
    prediction = correct_boundary_drift(
        prediction,
        masked,
        gap_index[0],
        gap_index[-1],
    )
    prediction = apply_physical_bounds(prediction, parameter)
    return prediction.reindex(gap_index), {
        "Production Detail": "xgboost_250_tree_rolling",
    }


def _production_donor_prediction(
    frames: dict[str, pd.DataFrame],
    station: str,
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    parameter: str,
) -> tuple[pd.Series, dict]:
    from VeryLongGaps import (
        apply_physical_bounds,
        choose_best_donor,
        correct_boundary_drift,
        donor_mean_prediction,
        fit_linear_map,
        min_std_for,
        prediction_from_donor,
    )

    donors = {
        sid: frame
        for sid, frame in frames.items()
        if sid != station and parameter in frame and frame[parameter].notna().any()
    }
    min_std = min_std_for(parameter)
    donor_sid, correlation, overlap = choose_best_donor(
        masked,
        {sid: frame[parameter] for sid, frame in donors.items()},
        min_overlap=1000,
        min_abs_corr=0.3,
        min_std=min_std,
    )
    prediction = pd.Series(np.nan, index=gap_index, dtype=float)
    detail = "donor_mean"
    if donor_sid is not None:
        model = fit_linear_map(masked, donors[donor_sid][parameter])
        linear = prediction_from_donor(
            gap_index,
            donors[donor_sid],
            parameter,
            model,
        )
        prediction.loc[linear.index] = linear
        detail = f"linear_donor={donor_sid};abs_corr={correlation:.4f};overlap={overlap}"
    fallback_index = gap_index[prediction.isna()]
    if len(fallback_index):
        fallback = donor_mean_prediction(
            fallback_index,
            donors,
            parameter,
            min_std,
        )
        prediction.loc[fallback.index] = fallback
        if donor_sid is not None and len(fallback):
            detail += ";donor_mean_fallback"
    prediction = correct_boundary_drift(
        prediction.dropna(),
        masked,
        gap_index[0],
        gap_index[-1],
    )
    prediction = apply_physical_bounds(prediction, parameter).reindex(gap_index)
    return prediction, {
        "Donor Correlation": correlation,
        "Production Detail": detail,
    }


def production_adapter_prediction(
    method: str,
    frames: dict[str, pd.DataFrame],
    case: pd.Series,
) -> tuple[pd.Series, dict]:
    station = case["Station"]
    parameter = case["Parameter"]
    frame = frames[station]
    gap_index = pd.date_range(case["Start"], case["End"], freq="h")
    masked = frame[parameter].copy()
    masked.loc[gap_index] = np.nan
    if method == "interpolation":
        return _production_interpolation_prediction(masked, gap_index, parameter)
    if method == "sarimax":
        return _production_sarimax_prediction(frame, masked, gap_index, parameter)
    if method == "xgboost":
        return _production_xgboost_prediction(frame, masked, gap_index, parameter)
    if method == "donor_regression":
        return _production_donor_prediction(
            frames, station, masked, gap_index, parameter
        )
    raise ValueError(f"No production adapter for {method}")


def production_validator_metrics(
    raw_prediction: pd.Series,
    masked: pd.Series,
    gap_index: pd.DatetimeIndex,
    parameter: str,
    gap_class: str,
) -> dict:
    """Apply adoption checks aligned with the stage-validator thresholds."""
    family = family_for(parameter)
    low, high = PHYSICAL_BOUNDS[family]
    values = pd.to_numeric(raw_prediction.reindex(gap_index), errors="coerce").clip(low, high)
    if gap_class == "verylong":
        jump_limit = 0.12 if parameter.startswith("SWC_") else 8.0
        boundary_limit = 0.08 if parameter.startswith("SWC_") else 8.0
        context_days = 14
        context_margin = 12.0
    else:
        jump_limit = 0.08 if parameter.startswith("SWC_") else 8.0
        boundary_limit = 0.05 if parameter.startswith("SWC_") else 5.0
        context_days = 7 if gap_class in {"short", "medium"} else 14
        context_margin = 10.0 if gap_class in {"short", "medium"} else 12.0

    reasons = []
    if values.isna().any():
        reasons.append("filled_value_nan")
    bound_hits = int(((values <= low) | (values >= high)).sum())
    if bound_hits:
        reasons.append("hit_physical_bound")
    diffs = values.diff().abs().dropna()
    max_hourly_change = float(diffs.max()) if len(diffs) else np.nan
    if np.isfinite(max_hourly_change) and max_hourly_change > jump_limit:
        reasons.append("large_hourly_jump")

    left = masked.get(gap_index[0] - pd.Timedelta(hours=1), np.nan)
    right = masked.get(gap_index[-1] + pd.Timedelta(hours=1), np.nan)
    start_jump = abs(values.iloc[0] - left) if pd.notna(left) else np.nan
    end_jump = abs(values.iloc[-1] - right) if pd.notna(right) else np.nan
    if np.isfinite(start_jump) and start_jump > boundary_limit:
        reasons.append("large_start_boundary_jump")
    if np.isfinite(end_jump) and end_jump > boundary_limit:
        reasons.append("large_end_boundary_jump")

    context_min = context_max = np.nan
    if parameter.startswith("T_"):
        before = masked.loc[
            gap_index[0] - pd.Timedelta(days=context_days):
            gap_index[0] - pd.Timedelta(hours=1)
        ]
        after = masked.loc[
            gap_index[-1] + pd.Timedelta(hours=1):
            gap_index[-1] + pd.Timedelta(days=context_days)
        ]
        context = pd.concat([before, after]).dropna()
        if len(context) >= 24:
            context_min = float(context.min())
            context_max = float(context.max())
            if float(values.min()) < context_min - context_margin:
                reasons.append("below_local_temperature_context")
            if float(values.max()) > context_max + context_margin:
                reasons.append("above_local_temperature_context")
    return {
        "Validator Status": "accepted" if not reasons else "rejected",
        "Validator Reasons": ";".join(dict.fromkeys(reasons)),
        "Bound Hits": bound_hits,
        "Max Hourly Change": max_hourly_change,
        "Start Boundary Jump": start_jump,
        "End Boundary Jump": end_jump,
        "Context Min": context_min,
        "Context Max": context_max,
    }


def run_production_adapter_smoke(
    frames: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    checkpoint_path: Path | None = None,
    initial_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    completed = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        cached = pd.read_csv(checkpoint_path)
        valid_version = (
            "Production Adapter Version" in cached
            and cached["Production Adapter Version"].eq(
                PRODUCTION_ADAPTER_VERSION
            ).all()
        )
        if valid_version:
            rows = cached.to_dict("records")
            completed = set(zip(cached["Case ID"], cached["Adapter Role"]))
            print(f"Reused {len(rows)} production-adapter checkpoint rows.")
    elif initial_results is not None and not initial_results.empty:
        selected_ids = set(cases["Case ID"])
        seeded = initial_results[
            initial_results["Case ID"].isin(selected_ids)
            & initial_results["Production Adapter Version"].eq(
                PRODUCTION_ADAPTER_VERSION
            )
        ].copy()
        rows = seeded.to_dict("records")
        completed = set(zip(seeded["Case ID"], seeded["Adapter Role"]))
        print(f"Seeded {len(rows)} rows from the smoke-test checkpoint.")
    for number, (_, case) in enumerate(cases.iterrows(), start=1):
        station = case["Station"]
        parameter = case["Parameter"]
        gap_index = pd.date_range(case["Start"], case["End"], freq="h")
        truth = frames[station][parameter].reindex(gap_index)
        masked = frames[station][parameter].copy()
        masked.loc[gap_index] = np.nan
        for role, method in [
            ("current", case["Current Production Method"]),
            ("proposed", case["Proposed Method"]),
        ]:
            if (case["Case ID"], role) in completed:
                continue
            started = time.perf_counter()
            error = ""
            metadata = {}
            try:
                prediction, metadata = production_adapter_prediction(
                    method, frames, case
                )
            except Exception as exc:
                prediction = pd.Series(np.nan, index=gap_index)
                error = str(exc)
            runtime = time.perf_counter() - started
            scored = score_prediction(
                case,
                truth,
                prediction,
                masked.dropna(),
                method,
                runtime,
                donor_correlation=metadata.get("Donor Correlation", np.nan),
                converged=metadata.get("Converged"),
                error=error,
            )
            scored.update(
                production_validator_metrics(
                    prediction,
                    masked,
                    gap_index,
                    parameter,
                    case["Gap Class"],
                )
            )
            scored.update(
                {
                    "Production Adapter Version": PRODUCTION_ADAPTER_VERSION,
                    "Adapter Role": role,
                    "Production Detail": metadata.get("Production Detail", ""),
                }
            )
            rows.append(scored)
            print(
                f"Production adapter {number}/{len(cases)}: "
                f"{station} {parameter} {case['Gap Class']} {role} "
                f"({method}, {runtime:.1f}s)"
            )
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    return pd.DataFrame(rows)


def production_adapter_smoke_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in results.groupby("Case ID"):
        current = group[group["Adapter Role"].eq("current")].iloc[0]
        proposed = group[group["Adapter Role"].eq("proposed")].iloc[0]
        complete = current["Status"] == "ok" and proposed["Status"] == "ok"
        lower_error = bool(
            complete and proposed["NRMSE IQR"] < current["NRMSE IQR"]
        )
        validator_pass = proposed["Validator Status"] == "accepted"
        advance = complete and lower_error and validator_pass
        rows.append(
            {
                "Case ID": case_id,
                "Station": proposed["Station"],
                "Parameter": proposed["Parameter"],
                "Gap Class": proposed["Gap Class"],
                "Season": proposed["Season"],
                "Hours": proposed["Hours"],
                "Current Method": current["Method"],
                "Proposed Method": proposed["Method"],
                "Current NRMSE IQR": current["NRMSE IQR"],
                "Proposed NRMSE IQR": proposed["NRMSE IQR"],
                "Current Validator Status": current["Validator Status"],
                "Proposed Validator Status": proposed["Validator Status"],
                "Proposed Validator Reasons": proposed["Validator Reasons"],
                "Current Runtime Seconds": current["Runtime Seconds"],
                "Proposed Runtime Seconds": proposed["Runtime Seconds"],
                "Smoke Decision": (
                    "advance_to_four_season_exact_trial"
                    if advance
                    else "hold_candidate_after_smoke"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["Parameter", "Gap Class"]).reset_index(drop=True)


def write_production_adapter_smoke_reports(
    cases: pd.DataFrame,
    results: pd.DataFrame,
    report_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = production_adapter_smoke_summary(results)
    paths = {
        "cases": report_dir / "soil_production_adapter_smoke_cases.csv",
        "detail": report_dir / "soil_production_adapter_smoke_detail.csv",
        "summary": report_dir / "soil_production_adapter_smoke_summary.csv",
    }
    cases.to_csv(paths["cases"], index=False)
    results.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return summary, paths


def select_production_adapter_four_season_cases(
    targeted_cases: pd.DataFrame,
    smoke_summary: pd.DataFrame,
) -> pd.DataFrame:
    advanced = smoke_summary[
        smoke_summary["Smoke Decision"].eq("advance_to_four_season_exact_trial")
    ][["Parameter", "Gap Class"]]
    selected = targeted_cases.merge(
        advanced,
        on=["Parameter", "Gap Class"],
        how="inner",
    )
    selected["Production Adapter Version"] = PRODUCTION_ADAPTER_VERSION
    return selected.sort_values(
        ["Parameter", "Gap Class", "Season", "Station"]
    ).reset_index(drop=True)


def production_adapter_four_season_summary(
    results: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (parameter, gap_class), planned in cases.groupby(["Parameter", "Gap Class"]):
        case_ids = set(planned["Case ID"])
        group = results[results["Case ID"].isin(case_ids)]
        current = group[group["Adapter Role"].eq("current")]
        proposed = group[group["Adapter Role"].eq("proposed")]
        current_ok = current[current["Status"].eq("ok")]
        proposed_ok = proposed[proposed["Status"].eq("ok")]
        paired = current_ok[["Case ID", "NRMSE IQR"]].merge(
            proposed_ok[["Case ID", "NRMSE IQR", "Season"]],
            on="Case ID",
            suffixes=(" Current", " Proposed"),
        )
        proposed_wins = int(
            (paired["NRMSE IQR Proposed"] < paired["NRMSE IQR Current"]).sum()
        )
        season_wins = int(
            paired.loc[
                paired["NRMSE IQR Proposed"] < paired["NRMSE IQR Current"],
                "Season",
            ].nunique()
        )
        planned_cases = len(planned)
        complete = (
            planned_cases == 4
            and planned["Season"].nunique() == 4
            and len(current_ok) == 4
            and len(proposed_ok) == 4
        )
        current_mean = paired["NRMSE IQR Current"].mean()
        proposed_mean = paired["NRMSE IQR Proposed"].mean()
        error_gate = bool(
            complete
            and proposed_mean < current_mean
            and proposed_wins >= 3
            and season_wins >= 3
        )
        validator_gate = bool(
            len(proposed) == planned_cases
            and proposed["Validator Status"].eq("accepted").all()
        )
        confirmed = complete and error_gate and validator_gate
        if confirmed:
            decision = "ready_for_non_destructive_pipeline_trial"
        elif not complete:
            decision = "incomplete_exact_trial_retain_current"
        elif not error_gate:
            decision = "exact_error_gate_failed_retain_current"
        else:
            decision = "exact_validator_gate_failed_retain_current"
        rows.append(
            {
                "Parameter": parameter,
                "Gap Class": gap_class,
                "Current Method": planned["Current Production Method"].iat[0],
                "Proposed Method": planned["Proposed Method"].iat[0],
                "Planned Cases": planned_cases,
                "Current Successful Cases": len(current_ok),
                "Proposed Successful Cases": len(proposed_ok),
                "Current Mean NRMSE IQR": current_mean,
                "Proposed Mean NRMSE IQR": proposed_mean,
                "Proposed Case Wins": proposed_wins,
                "Proposed Season Wins": season_wins,
                "Current Runtime Seconds": current["Runtime Seconds"].sum(),
                "Proposed Runtime Seconds": proposed["Runtime Seconds"].sum(),
                "Current Timeout/Failure Cases": int((current["Status"] != "ok").sum()),
                "Proposed Validator Rejections": int(
                    (proposed["Validator Status"] != "accepted").sum()
                ),
                "Complete Four-Season Exact Trial": complete,
                "Exact Error Gate": error_gate,
                "Exact Validator Gate": validator_gate,
                "Exact Trial Decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values(["Parameter", "Gap Class"]).reset_index(drop=True)


def write_production_adapter_four_season_reports(
    cases: pd.DataFrame,
    results: pd.DataFrame,
    report_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = production_adapter_four_season_summary(results, cases)
    paths = {
        "cases": report_dir / "soil_production_adapter_four_season_cases.csv",
        "detail": report_dir / "soil_production_adapter_four_season_detail.csv",
        "summary": report_dir / "soil_production_adapter_four_season_summary.csv",
    }
    cases.to_csv(paths["cases"], index=False)
    results.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return summary, paths


def final_production_method_map(
    benchmark_map: pd.DataFrame,
    smoke_summary: pd.DataFrame,
    exact_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve all benchmark candidates without changing production scripts."""
    smoke = smoke_summary[
        ["Parameter", "Gap Class", "Smoke Decision"]
    ]
    exact = exact_summary[
        ["Parameter", "Gap Class", "Exact Trial Decision"]
    ]
    table = benchmark_map.merge(
        smoke,
        on=["Parameter", "Gap Class"],
        how="left",
    ).merge(
        exact,
        on=["Parameter", "Gap Class"],
        how="left",
    )
    ready = table["Exact Trial Decision"].eq(
        "ready_for_non_destructive_pipeline_trial"
    )
    table["Final Production Method"] = table["Current Production Method"]
    table["Next Pipeline Trial Method"] = table["Current Production Method"]
    table.loc[ready, "Next Pipeline Trial Method"] = table.loc[
        ready, "Proposed Method"
    ]
    table["Production Adapter Decision"] = np.select(
        [
            ready,
            table["Exact Trial Decision"].notna(),
            table["Smoke Decision"].eq("hold_candidate_after_smoke"),
            table["Final Recommendation"].eq("candidate_for_production_trial"),
        ],
        [
            "ready_for_non_destructive_pipeline_trial",
            "retain_current_after_four_season_exact_trial",
            "retain_current_after_exact_smoke",
            "retain_current_exact_evidence_incomplete",
        ],
        default="retain_current_no_adapter_change_requested",
    )
    table["Production Script Changed"] = False
    return table.sort_values(["Parameter", "Gap Class"]).reset_index(drop=True)


def production_adapter_decision_figure(
    benchmark_map: pd.DataFrame,
    smoke_summary: pd.DataFrame,
    exact_summary: pd.DataFrame,
    report_dir: Path,
) -> tuple[plt.Figure, Path, Path]:
    """Summarize the selection funnel and exact four-season comparisons."""
    stages = [
        ("Stable seed-and-season winners", int(benchmark_map["Stable Across Seeds and Seasons"].sum())),
        ("Winners different from current map", int(benchmark_map["Requires Targeted Confirmation"].sum())),
        ("Independent confirmations", int(benchmark_map["Final Recommendation"].eq("candidate_for_production_trial").sum())),
        ("Exact smoke advances", int(smoke_summary["Smoke Decision"].eq("advance_to_four_season_exact_trial").sum())),
        ("Four-season exact advances", int(exact_summary["Exact Trial Decision"].eq("ready_for_non_destructive_pipeline_trial").sum())),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={"width_ratios": [0.9, 1.35]})

    labels = [item[0] for item in stages][::-1]
    counts = [item[1] for item in stages][::-1]
    colors = ["#C62828" if value == 0 else "#28666E" for value in counts]
    y = np.arange(len(labels))
    axes[0].barh(y, counts, color=colors, height=0.62)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0, max(counts) * 1.18)
    axes[0].set_xlabel("Parameter-gap combinations")
    axes[0].set_title("A. Selection funnel", fontweight="bold")
    axes[0].grid(axis="x", color="#DDDDDD", linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    for position, value in enumerate(counts):
        axes[0].text(value + 0.6, position, str(value), va="center", fontweight="bold")

    exact = exact_summary.copy()
    exact["Parameter Order"] = exact["Parameter"].map(
        {name: number for number, name in enumerate(SOIL_PARAMETERS)}
    )
    exact["Gap Order"] = exact["Gap Class"].map(
        {name: number for number, name in enumerate(GAP_CLASSES)}
    )
    exact = exact.sort_values(["Parameter Order", "Gap Order"])
    y = np.arange(len(exact))
    incomplete = ~exact["Complete Four-Season Exact Trial"]
    line_colors = np.where(incomplete, "#E08E0B", "#C62828")
    for position, (_, row) in enumerate(exact.iterrows()):
        axes[1].plot(
            [row["Current Mean NRMSE IQR"], row["Proposed Mean NRMSE IQR"]],
            [position, position],
            color=line_colors[position],
            linewidth=2.2,
        )
    axes[1].scatter(
        exact["Current Mean NRMSE IQR"], y,
        marker="s", s=55, color="#777777", zorder=3,
    )
    axes[1].scatter(
        exact["Proposed Mean NRMSE IQR"], y,
        marker="o", s=62, c=line_colors, zorder=4,
    )
    exact_labels = []
    for _, row in exact.iterrows():
        parameter = (
            f"SWC {row['Parameter'].removeprefix('SWC_')} cm"
            if row["Parameter"].startswith("SWC_")
            else f"Soil T {row['Parameter'].removeprefix('T_')} cm"
        )
        suffix = "*" if not row["Complete Four-Season Exact Trial"] else ""
        exact_labels.append(f"{parameter} | {row['Gap Class'].title()}{suffix}")
    axes[1].set_yticks(y, exact_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean NRMSE / observed IQR")
    axes[1].set_title("B. Exact production-adapter comparison", fontweight="bold")
    axes[1].grid(axis="x", color="#DDDDDD", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].text(
        0.0, -0.17,
        "* Incomplete: two of four current auto-SARIMAX fits exceeded 180 s.",
        transform=axes[1].transAxes,
        fontsize=9,
    )
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777",
               markeredgecolor="#777777", label="Current production method"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C62828",
               markeredgecolor="#C62828", label="Proposed method"),
        Line2D([0], [0], color="#C62828", linewidth=2.5, label="Retain current"),
        Line2D([0], [0], color="#E08E0B", linewidth=2.5, label="Incomplete exact trial"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "TxSON Soil Method Selection With Exact Production Adapters",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.20, right=0.98, wspace=0.48)
    report_dir.mkdir(parents=True, exist_ok=True)
    png = report_dir / "soil_production_adapter_decision_publication.png"
    pdf = report_dir / "soil_production_adapter_decision_publication.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return fig, png, pdf

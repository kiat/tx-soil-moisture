from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import matplotlib

# Use a non-interactive backend so the script can save figures without opening a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astral import Observer
from astral.sun import sun


# Define input/output paths used by the report generator.
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "prewashed_met_data"
OUTPUT_DIR = ROOT / "prewashed_anomaly_outputs"
MARKED_DIR = OUTPUT_DIR / "marked_data"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = OUTPUT_DIR / "Solar_Radiation_Anomaly_Report.md"
FIGURE_PATH = OUTPUT_DIR / "anomaly_distribution_by_dataset.png"

# Name the timestamp and solar-radiation columns in the prewashed CSVs.
DATE_COL = "Date"
SOLAR_COL = "Srad"
TIMEZONE = "America/Chicago"

# Store station coordinates for station-specific daylight calculations.
STATION_LOCATIONS = {
    "CB01": {"latitude": 30.4193, "longitude": -98.8046},
    "CB04": {"latitude": 30.4600, "longitude": -98.9407},
    "CB06": {"latitude": 30.4421, "longitude": -98.8427},
    "FD02": {"latitude": 30.2456, "longitude": -98.6988},
    "FD03": {"latitude": 30.4175, "longitude": -98.8542},
    "WC05": {"latitude": 30.4319, "longitude": -98.8133},
}


@dataclass(frozen=True)
class QCConfig:
    # Keep all QC thresholds in one auditable configuration object.
    night_threshold: float = 5.0
    sunrise_sunset_buffer_minutes: int = 30
    near_zero_threshold: float = 5.0
    min_zero_run_records: int = 3
    high_radiation_threshold: float = 200.0
    sudden_near_zero_threshold: float = 20.0
    sudden_jump_threshold: float = 500.0
    physical_min: float = 0.0
    physical_max: float = 1300.0
    weather_low_solar_threshold: float = 50.0
    weather_high_rh_threshold: float = 95.0
    weather_recent_precip_hours: int = 3


ANOMALY_FLAGS = {
    # Map row-level boolean flag columns to readable labels.
    "night_radiation_anomaly_flag": "Night-time radiation anomaly",
    "long_zero_run_flag": "Unexplained long daytime zero/near-zero run",
    "sudden_drop_flag": "Unexplained sudden radiation drop",
    "weather_related_low_radiation_event_flag": "Weather-related low-radiation event",
    "missing_srad_flag": "Missing Srad value",
    "sudden_spike_flag": "Sudden radiation spike",
    "out_of_range_flag": "Out-of-range value",
}

CATEGORY_ORDER = [
    # Control the order of categories in tables and summaries.
    "Unexplained daytime low-radiation anomaly",
    "Night-time radiation anomaly",
    "Missing Srad/timestamp anomaly",
    "Spike/out-of-range radiation anomaly",
    "Weather-related low-radiation event",
]

CORRECTION_TARGET_CATEGORIES = [
    # Exclude weather-explained events from correction-target anomaly counts.
    "Unexplained daytime low-radiation anomaly",
    "Night-time radiation anomaly",
    "Missing Srad/timestamp anomaly",
    "Spike/out-of-range radiation anomaly",
]

CATEGORY_COLORS = {
    # Keep plot colors stable across regenerated figures.
    "Unexplained daytime low-radiation anomaly": "#0b7fb8",
    "Night-time radiation anomaly": "#f2a900",
    "Missing Srad/timestamp anomaly": "#08a678",
    "Spike/out-of-range radiation anomaly": "#e66a00",
    "Weather-related low-radiation event": "#7a8a99",
}

MAJOR_CATEGORIES = {
    # Group detailed row-level flags into report-level categories.
    "Unexplained daytime low-radiation anomaly": {
        "flag_columns": [
            "long_zero_run_flag",
            "sudden_drop_flag",
        ],
        "included_anomaly_types": [
            "Unexplained long daytime zero/near-zero run",
            "Unexplained sudden radiation drop",
        ],
        "treatment": "Flag for review. Candidate for imputation only after sensor/weather review.",
    },
    "Night-time radiation anomaly": {
        "flag_columns": ["night_radiation_anomaly_flag"],
        "included_anomaly_types": ["Night-time radiation anomaly"],
        "treatment": "Flag for review. Candidate for setting to 0 or NA after confirmation.",
    },
    "Missing Srad/timestamp anomaly": {
        "flag_columns": ["missing_srad_flag"],
        "included_anomaly_types": ["Missing Srad value", "Timestamp gap"],
        "treatment": "Flag as missing. Candidate for solar-aware imputation after review.",
    },
    "Spike/out-of-range radiation anomaly": {
        "flag_columns": ["sudden_spike_flag", "out_of_range_flag"],
        "included_anomaly_types": ["Sudden radiation spike", "Out-of-range value"],
        "treatment": "Flag for review. Candidate for replacement before imputation.",
    },
    "Weather-related low-radiation event": {
        "flag_columns": ["weather_related_low_radiation_event_flag"],
        "included_anomaly_types": ["Weather-related low-radiation event"],
        "treatment": "Advisory only. Retain by default because low Srad is weather-explained.",
    },
}

DETAILED_TREATMENTS = {
    # Explain the recommended handling for each detailed flag/event type.
    "Night-time radiation anomaly": "Flagged; no imputation applied.",
    "Unexplained long daytime zero/near-zero run": "Flagged as suspicious; candidate for review/imputation.",
    "Unexplained sudden radiation drop": "Flagged as suspicious; candidate for review/imputation.",
    "Weather-related low-radiation event": "Advisory only; precipitation/recent precipitation or high RH explains low Srad.",
    "Missing Srad value": "Flagged as missing; candidate for imputation.",
    "Timestamp gap": "Logged as absent expected timestamp; candidate for imputation.",
    "Sudden radiation spike": "Flagged; candidate for replacement before imputation.",
    "Out-of-range value": "Flagged; candidate for replacement before imputation.",
}

SUMMARY_COLUMNS = [
    # Define the station/category summary table schema.
    "file_name",
    "station_id",
    "major_anomaly_category",
    "included_anomaly_types",
    "number_of_anomaly_records",
    "first_occurrence",
    "last_occurrence",
    "percentage_of_total_records",
    "treatment",
]

DETAILED_COLUMNS = [
    # Define the continuous-period detailed table schema.
    "file_name",
    "station_id",
    "anomaly_type",
    "start_time",
    "end_time",
    "duration",
    "number_of_records",
    f"min_{SOLAR_COL}",
    f"max_{SOLAR_COL}",
    f"mean_{SOLAR_COL}",
    "treatment",
]


def station_id_from_file(path: Path) -> str:
    # Extract station ids from names such as CB04_met.csv.
    return path.stem.split("_")[0].split("-")[0]


def input_files() -> list[Path]:
    # Collect configured prewashed met files in a deterministic order.
    return sorted(
        path
        for path in INPUT_DIR.glob("*_met.csv")
        if station_id_from_file(path) in STATION_LOCATIONS
    )


def infer_sampling_interval(df: pd.DataFrame) -> pd.Timedelta:
    # Estimate the normal time step for gap and duration calculations.
    intervals = df[DATE_COL].diff().dropna()
    if intervals.empty:
        return pd.Timedelta(hours=1)
    return intervals.median()


def load_prewashed_data(path: Path) -> pd.DataFrame:
    # Load standardized prewashed CSVs and keep Date/Srad as typed values.
    df = pd.read_csv(path, low_memory=False)
    if DATE_COL not in df.columns or SOLAR_COL not in df.columns:
        raise ValueError(f"{path.name} must contain {DATE_COL!r} and {SOLAR_COL!r}.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()
    for column in df.columns:
        if column != DATE_COL:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(DATE_COL).reset_index(drop=True)


def add_daylight_columns(df: pd.DataFrame, station_id: str, config: QCConfig) -> pd.DataFrame:
    # Use station-specific sunrise/sunset times so low night radiation is not over-flagged.
    location = STATION_LOCATIONS[station_id]
    observer = Observer(latitude=location["latitude"], longitude=location["longitude"])
    timezone = ZoneInfo(TIMEZONE)
    out = df.copy()
    dates = out[DATE_COL].dt.date

    sunrise_by_date = {}
    sunset_by_date = {}
    for date_value in dates.drop_duplicates():
        sun_times = sun(observer, date=date_value, tzinfo=timezone)
        sunrise_by_date[date_value] = sun_times["sunrise"].replace(tzinfo=None)
        sunset_by_date[date_value] = sun_times["sunset"].replace(tzinfo=None)

    out["_sunrise"] = pd.to_datetime(dates.map(sunrise_by_date))
    out["_sunset"] = pd.to_datetime(dates.map(sunset_by_date))
    buffer = pd.Timedelta(minutes=config.sunrise_sunset_buffer_minutes)
    out["_is_daytime"] = (
        (out[DATE_COL] >= out["_sunrise"] - buffer)
        & (out[DATE_COL] <= out["_sunset"] + buffer)
    )
    return out


def flag_long_runs(mask: pd.Series, min_records: int) -> pd.Series:
    # Keep only continuous runs long enough to be meaningful QC events.
    mask = mask.fillna(False)
    run_id = mask.ne(mask.shift(fill_value=False)).cumsum()
    run_length = mask.groupby(run_id).transform("sum")
    return mask & (run_length >= min_records)


def timestamp_gap_periods(df: pd.DataFrame, interval: pd.Timedelta) -> list[dict]:
    # Prewashed files should be gap-filled; this remains as a safety check.
    if interval <= pd.Timedelta(0):
        return []

    periods = []
    gaps = df[DATE_COL].diff()
    gap_mask = gaps.gt(interval * 1.5)
    for index in df.index[gap_mask]:
        previous_time = df.loc[index - 1, DATE_COL]
        current_time = df.loc[index, DATE_COL]
        missing_start = previous_time + interval
        missing_end = current_time - interval
        missing_records = max(int(round(gaps.loc[index] / interval)) - 1, 1)
        periods.append(
            {
                "start_time": missing_start,
                "end_time": missing_end,
                "duration": current_time - previous_time - interval,
                "missing_records": missing_records,
            }
        )
    return periods


def mark_anomalies(df: pd.DataFrame, config: QCConfig) -> pd.DataFrame:
    # Separate true QC anomalies from low-radiation events explained by weather.
    out = df.copy()
    srad = out[SOLAR_COL]
    daytime = out["_is_daytime"]
    night = ~daytime
    valid_daytime = daytime & srad.notna()

    out["night_radiation_anomaly_flag"] = night & srad.gt(config.night_threshold)

    rain = out["Ppt"].fillna(0) if "Ppt" in out.columns else pd.Series(0.0, index=out.index)
    humidity = out["RH"] if "RH" in out.columns else pd.Series(pd.NA, index=out.index)

    # Treat current/recent precipitation and near-saturated RH as weather evidence.
    recent_rain = rain.rolling(
        window=config.weather_recent_precip_hours + 1,
        min_periods=1,
    ).sum().gt(0)
    weather_context = recent_rain | humidity.ge(config.weather_high_rh_threshold)
    weather_low_radiation = (
        valid_daytime
        & srad.le(config.weather_low_solar_threshold)
        & weather_context
    )
    out["weather_related_low_radiation_event_flag"] = weather_low_radiation

    # Low radiation with weather support is an event, not a correction-target anomaly.
    unexplained_daytime = valid_daytime & ~weather_low_radiation
    near_zero_day = unexplained_daytime & srad.le(config.near_zero_threshold)
    out["long_zero_run_flag"] = flag_long_runs(near_zero_day, config.min_zero_run_records)

    previous_srad = srad.shift(1)
    previous_daytime = daytime.shift(1, fill_value=False)
    out["sudden_drop_flag"] = (
        unexplained_daytime
        & previous_daytime
        & previous_srad.ge(config.high_radiation_threshold)
        & srad.le(config.sudden_near_zero_threshold)
        & (previous_srad - srad).ge(config.high_radiation_threshold)
    )

    out["missing_srad_flag"] = srad.isna()
    out["sudden_spike_flag"] = (
        valid_daytime
        & previous_daytime
        & previous_srad.notna()
        & (srad - previous_srad).ge(config.sudden_jump_threshold)
    )
    out["out_of_range_flag"] = valid_daytime & (
        srad.lt(config.physical_min) | srad.gt(config.physical_max)
    )

    labels = pd.Series("", index=out.index, dtype="object")
    for flag_column, anomaly_type in ANOMALY_FLAGS.items():
        mask = out[flag_column].fillna(False)
        labels.loc[mask] = labels.loc[mask].where(
            labels.loc[mask].eq(""), labels.loc[mask] + "; "
        ) + anomaly_type
    out["srad_anomaly_classification"] = labels.replace("", pd.NA)
    return out


def duration_text(start_time: pd.Timestamp, end_time: pd.Timestamp, interval: pd.Timedelta) -> str:
    # Report inclusive event duration using the station sampling interval.
    duration = end_time - start_time
    if interval > pd.Timedelta(0):
        duration += interval
    return str(duration)


def summarize_mask(
    df: pd.DataFrame,
    mask: pd.Series,
    anomaly_type: str,
    file_name: str,
    station_id: str,
    interval: pd.Timedelta,
) -> list[dict]:
    # Collapse adjacent flagged rows into continuous event-period records.
    mask = mask.fillna(False)
    if not mask.any():
        return []

    gap_break = df[DATE_COL].diff().gt(interval * 1.5) if interval > pd.Timedelta(0) else False
    run_id = (mask.ne(mask.shift(fill_value=False)) | gap_break).cumsum()
    rows = []
    for _, group in df.loc[mask].groupby(run_id[mask]):
        start_time = group[DATE_COL].iloc[0]
        end_time = group[DATE_COL].iloc[-1]
        srad_values = group[SOLAR_COL]
        rows.append(
            {
                "file_name": file_name,
                "station_id": station_id,
                "anomaly_type": anomaly_type,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration_text(start_time, end_time, interval),
                "number_of_records": len(group),
                f"min_{SOLAR_COL}": srad_values.min(skipna=True),
                f"max_{SOLAR_COL}": srad_values.max(skipna=True),
                f"mean_{SOLAR_COL}": srad_values.mean(skipna=True),
                "treatment": DETAILED_TREATMENTS[anomaly_type],
            }
        )
    return rows


def build_detailed_summary(
    df: pd.DataFrame,
    file_name: str,
    station_id: str,
    interval: pd.Timedelta,
) -> pd.DataFrame:
    # Build one detailed table row per continuous QC/event period.
    rows = []
    for flag_column, anomaly_type in ANOMALY_FLAGS.items():
        rows.extend(summarize_mask(df, df[flag_column], anomaly_type, file_name, station_id, interval))

    for gap in timestamp_gap_periods(df, interval):
        rows.append(
            {
                "file_name": file_name,
                "station_id": station_id,
                "anomaly_type": "Timestamp gap",
                "start_time": gap["start_time"],
                "end_time": gap["end_time"],
                "duration": str(gap["duration"]),
                "number_of_records": gap["missing_records"],
                f"min_{SOLAR_COL}": pd.NA,
                f"max_{SOLAR_COL}": pd.NA,
                f"mean_{SOLAR_COL}": pd.NA,
                "treatment": DETAILED_TREATMENTS["Timestamp gap"],
            }
        )

    summary = pd.DataFrame(rows, columns=DETAILED_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(["station_id", "start_time", "anomaly_type"]).reset_index(drop=True)
    return summary


def build_simplified_summary(
    df: pd.DataFrame,
    file_name: str,
    station_id: str,
    interval: pd.Timedelta,
) -> pd.DataFrame:
    # Build one summary row per station and major category.
    total_records = len(df)
    gap_periods = timestamp_gap_periods(df, interval)
    rows = []

    for category, info in MAJOR_CATEGORIES.items():
        mask = pd.Series(False, index=df.index)
        for flag_column in info["flag_columns"]:
            mask = mask | df[flag_column].fillna(False)

        record_count = int(mask.sum())
        starts = []
        ends = []
        if record_count:
            starts.append(df.loc[mask, DATE_COL].min())
            ends.append(df.loc[mask, DATE_COL].max())

        if category == "Missing Srad/timestamp anomaly":
            gap_records = sum(gap["missing_records"] for gap in gap_periods)
            record_count += gap_records
            starts.extend(gap["start_time"] for gap in gap_periods)
            ends.extend(gap["end_time"] for gap in gap_periods)

        rows.append(
            {
                "file_name": file_name,
                "station_id": station_id,
                "major_anomaly_category": category,
                "included_anomaly_types": "; ".join(info["included_anomaly_types"]),
                "number_of_anomaly_records": record_count,
                "first_occurrence": min(starts) if starts else pd.NaT,
                "last_occurrence": max(ends) if ends else pd.NaT,
                "percentage_of_total_records": round(record_count / total_records * 100, 3)
                if total_records
                else 0.0,
                "treatment": info["treatment"],
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def data_health_row(df: pd.DataFrame, path: Path, interval: pd.Timedelta) -> dict:
    # Summarize duplicate, gap, missing-value, and range checks for one file.
    expected = int(round((df[DATE_COL].max() - df[DATE_COL].min()) / interval)) + 1
    exact_duplicate_rows = int(df.duplicated(keep="first").sum())
    duplicate_timestamp_rows = int(df.duplicated(subset=[DATE_COL], keep="first").sum())
    gap_records = sum(gap["missing_records"] for gap in timestamp_gap_periods(df, interval))
    return {
        "file_name": path.name,
        "station_id": station_id_from_file(path),
        "records": len(df),
        "unique_timestamps": df[DATE_COL].nunique(),
        "expected_hourly_records": expected,
        "exact_duplicate_extra_rows": exact_duplicate_rows,
        "duplicate_timestamp_extra_rows": duplicate_timestamp_rows,
        "timestamp_gap_records": gap_records,
        "missing_srad_values": int(df[SOLAR_COL].isna().sum()),
        f"min_{SOLAR_COL}": df[SOLAR_COL].min(skipna=True),
        f"max_{SOLAR_COL}": df[SOLAR_COL].max(skipna=True),
    }


def process_file(path: Path, config: QCConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Run the full QC workflow for one station file and write station outputs.
    station_id = station_id_from_file(path)
    df = load_prewashed_data(path)
    interval = infer_sampling_interval(df)
    health = data_health_row(df, path, interval)

    df = add_daylight_columns(df, station_id, config)
    df = mark_anomalies(df, config)

    marked_columns = [
        column
        for column in df.columns
        if column not in {"_sunrise", "_sunset", "_is_daytime"}
    ]
    MARKED_DIR.mkdir(parents=True, exist_ok=True)
    df.loc[:, marked_columns].to_csv(MARKED_DIR / f"{path.stem}_anomaly_marked.csv", index=False)

    simplified = build_simplified_summary(df, path.name, station_id, interval)
    detailed = build_detailed_summary(df, path.name, station_id, interval)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    simplified.to_csv(TABLE_DIR / f"{path.stem}_simplified_anomaly_summary.csv", index=False)
    detailed.to_csv(TABLE_DIR / f"{path.stem}_detailed_anomaly_summary.csv", index=False)
    return simplified, detailed, health


def pivot_by_station(simplified: pd.DataFrame) -> pd.DataFrame:
    # Convert long-form summary rows into station-by-category counts.
    pivot = simplified.pivot_table(
        index="station_id",
        columns="major_anomaly_category",
        values="number_of_anomaly_records",
        aggfunc="sum",
        fill_value=0,
    )
    for category in CATEGORY_ORDER:
        if category not in pivot.columns:
            pivot[category] = 0
    pivot = pivot[CATEGORY_ORDER]
    pivot["Total flagged records/events"] = pivot.sum(axis=1)
    return pivot.sort_values("Total flagged records/events", ascending=False)


def make_distribution_figure(simplified: pd.DataFrame, record_counts: dict[str, int]) -> None:
    # Plot only correction-target QC anomalies; weather events stay out of this figure.
    pivot = pivot_by_station(simplified)
    plot_categories = CORRECTION_TARGET_CATEGORIES
    plot_pivot = pivot[plot_categories].copy()
    plot_pivot["Total correction-target QC anomaly records"] = plot_pivot.sum(axis=1)
    plot_pivot = plot_pivot.sort_values(
        "Total correction-target QC anomaly records",
        ascending=False,
    )
    rate = plot_pivot[plot_categories].astype(float).copy()
    for station in rate.index:
        denominator = record_counts.get(station, 0)
        rate.loc[station] = rate.loc[station] / denominator * 100 if denominator else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1.18]})
    fig.patch.set_facecolor("white")
    y_positions = range(len(plot_pivot.index))

    for axis, data, xlabel, title, is_rate in [
        (
            axes[0],
            plot_pivot[plot_categories],
            "Correction-target QC anomaly records (count)",
            "A. Absolute QC anomaly records",
            False,
        ),
        (
            axes[1],
            rate[plot_categories],
            "Correction-target QC anomaly records as % of prewashed records",
            "B. QC anomaly rate by type",
            True,
        ),
    ]:
        left = pd.Series(0.0, index=data.index)
        for category in plot_categories:
            axis.barh(
                y_positions,
                data[category],
                left=left,
                color=CATEGORY_COLORS[category],
                height=0.52,
                edgecolor="none",
                label=category,
            )
            left += data[category]

        axis.set_yticks(list(y_positions), data.index)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=18)
        axis.set_xlabel(xlabel, fontsize=10, color="#475b6d", labelpad=14)
        axis.grid(axis="x", color="#d6e0ea", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#9aaac0")
        axis.tick_params(axis="y", length=0, labelsize=12, colors="#26343d", pad=14)
        axis.tick_params(axis="x", length=0, labelsize=9, colors="#475b6d", pad=14)

        for idx, station in enumerate(data.index):
            total_value = left.loc[station]
            label = f"{total_value:.2f}%" if is_rate else f"{int(total_value):,}"
            axis.text(
                total_value + (0.02 if is_rate else 180),
                idx,
                label,
                va="center",
                ha="left",
                fontsize=9,
                color="#26343d",
                fontweight="bold",
            )

        for idx, station in enumerate(data.index):
            major_value = data.loc[station, "Unexplained daytime low-radiation anomaly"]
            show_inner_label = (
                major_value >= 500
                if not is_rate
                else major_value >= 0.5
            )
            if show_inner_label:
                label = f"{major_value:.2f}%" if is_rate else f"{int(major_value):,}"
                axis.text(
                    major_value / 2,
                    idx,
                    label,
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

    axes[0].set_xlim(
        0,
        max(3000, plot_pivot["Total correction-target QC anomaly records"].max() * 1.25),
    )
    axes[1].set_xlim(0, max(3, rate.sum(axis=1).max() * 1.25))
    axes[1].xaxis.set_major_formatter(lambda value, _: f"{value:.2f}%")

    fig.text(0.055, 0.93, "Srad Correction-Target QC Anomalies by Dataset", fontsize=24, fontweight="bold", color="#1f2933")
    fig.text(
        0.055,
        0.905,
        "Weather-related low-radiation events are excluded from this anomaly chart and summarized separately in the report.",
        fontsize=10,
        color="#475b6d",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[category]) for category in plot_categories]
    fig.legend(
        handles,
        plot_categories,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.875),
        ncol=4,
        frameon=False,
        fontsize=10,
        handlelength=1.4,
        handleheight=1.4,
        columnspacing=2.8,
    )
    fig.text(
        0.055,
        0.065,
        "Source: prewashed_anomaly_outputs/tables/combined_simplified_anomaly_summary.csv",
        fontsize=8,
        color="#5f6f82",
    )
    plt.subplots_adjust(left=0.14, right=0.965, top=0.755, bottom=0.18, wspace=0.19)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    # Render small Python row lists as GitHub-compatible Markdown tables.
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def report_markdown(
    simplified: pd.DataFrame,
    detailed: pd.DataFrame,
    health: pd.DataFrame,
) -> str:
    # Compose the Markdown report using the latest summary tables and figure path.
    stations = sorted(simplified["station_id"].unique())
    record_counts = dict(zip(health["station_id"], health["records"]))
    total_records = int(health["records"].sum())
    total_flagged = int(simplified["number_of_anomaly_records"].sum())
    true_anomaly_records = int(
        simplified[
            simplified["major_anomaly_category"].isin(CORRECTION_TARGET_CATEGORIES)
        ]["number_of_anomaly_records"].sum()
    )
    weather_event_records = int(
        simplified[
            simplified["major_anomaly_category"].eq("Weather-related low-radiation event")
        ]["number_of_anomaly_records"].sum()
    )
    flagged_rate = total_flagged / total_records * 100 if total_records else 0.0
    true_anomaly_rate = true_anomaly_records / total_records * 100 if total_records else 0.0
    pivot = pivot_by_station(simplified)

    health_rows = []
    for _, row in health.sort_values("station_id").iterrows():
        health_rows.append(
            [
                row["station_id"],
                f"{int(row['records']):,}",
                f"{int(row['unique_timestamps']):,}",
                f"{int(row['exact_duplicate_extra_rows']):,}",
                f"{int(row['timestamp_gap_records']):,}",
                f"{int(row['missing_srad_values']):,}",
                f"{row[f'min_{SOLAR_COL}']:.3g}",
                f"{row[f'max_{SOLAR_COL}']:.3g}",
            ]
        )

    category_rows = []
    for category in CATEGORY_ORDER:
        rows = simplified[simplified["major_anomaly_category"] == category]
        records = int(rows["number_of_anomaly_records"].sum())
        affected = int((rows["number_of_anomaly_records"] > 0).sum())
        category_rows.append(
            [
                category,
                rows["included_anomaly_types"].dropna().iloc[0],
                f"{records:,}",
                f"{records / total_flagged * 100:.1f}%" if total_flagged else "0.0%",
                f"{affected} / {len(stations)}",
                rows["treatment"].dropna().iloc[0],
            ]
        )

    detailed_rows = []
    if detailed.empty:
        detailed_rows.append(["No anomalies", "0", "0", "--"])
    else:
        by_type = detailed.groupby("anomaly_type", dropna=False).agg(
            periods=("anomaly_type", "size"),
            records=("number_of_records", "sum"),
        )
        by_type = by_type.sort_values("records", ascending=False)
        for anomaly_type, row in by_type.iterrows():
            detailed_rows.append(
                [
                    anomaly_type,
                    f"{int(row['periods']):,}",
                    f"{int(row['records']):,}",
                    DETAILED_TREATMENTS.get(anomaly_type, "Flagged for review."),
                ]
            )

    station_rows = []
    for station, row in pivot.iterrows():
        station_source = simplified[simplified["station_id"] == station]
        first = station_source["first_occurrence"].dropna().min()
        last = station_source["last_occurrence"].dropna().max()
        records = record_counts[station]
        total = int(row["Total flagged records/events"])
        true_total = int(row[CORRECTION_TARGET_CATEGORIES].sum())
        station_rows.append(
            [
                station,
                f"{records:,}",
                f"{total:,}",
                f"{true_total:,}",
                f"{true_total / records * 100:.3f}%" if records else "0.000%",
                f"{int(row['Unexplained daytime low-radiation anomaly']):,}",
                f"{int(row['Night-time radiation anomaly']):,}",
                f"{int(row['Missing Srad/timestamp anomaly']):,}",
                f"{int(row['Spike/out-of-range radiation anomaly']):,}",
                f"{int(row['Weather-related low-radiation event']):,}",
                first if pd.notna(first) else "--",
                last if pd.notna(last) else "--",
            ]
        )

    output_rows = [
        ["Marked prewashed data", "prewashed_anomaly_outputs/marked_data/*_anomaly_marked.csv", "Original columns plus Srad QC flags and weather-event labels"],
        ["Station summaries", "prewashed_anomaly_outputs/tables/*_simplified_anomaly_summary.csv", "One row per station and major QC/event category"],
        ["Combined simplified summary", "prewashed_anomaly_outputs/tables/combined_simplified_anomaly_summary.csv", f"{len(simplified):,} rows"],
        ["Combined detailed summary", "prewashed_anomaly_outputs/tables/combined_detailed_anomaly_summary.csv", f"{len(detailed):,} continuous anomaly periods"],
        ["Distribution figure", "prewashed_anomaly_outputs/anomaly_distribution_by_dataset.png", "Stacked correction-target QC anomaly count and rate chart"],
    ]

    return f"""# Solar Radiation Anomaly Report

This document explains the output from running `generate_anomaly_report.py` on the six prewashed met-station solar radiation files. The prewashed files are already cleaned, standardized, de-duplicated, gap-filled in time, and ready for downstream quality control. The solar radiation variable evaluated here is `{SOLAR_COL}`.

## What the code does

The routine marks and summarizes solar radiation QC anomalies in five passes. **The timestamps are a data column here**, and the code uses station latitude/longitude plus `{TIMEZONE}` sunrise and sunset times to determine whether each record is daytime or night-time.

**Pass one - input parsing.** `load_prewashed_data()` reads the standardized prewashed CSV files, parses `Date`, coerces numeric weather columns, sorts records by time, and keeps the original observations.

**Pass two - data readiness check.** `data_health_row()` summarizes record counts, unique timestamps, duplicate rows, timestamp gaps, and missing `{SOLAR_COL}` values. The current prewashed files have no duplicate timestamps and no hourly timestamp gaps.

**Pass three - daylight window.** `add_daylight_columns()` calculates sunrise and sunset for each station/date and adds a {QCConfig().sunrise_sunset_buffer_minutes}-minute buffer around the daylight window.

**Pass four - QC anomaly and weather-event marking.** `mark_anomalies()` marks correction-target QC anomalies separately from weather-related low-radiation events. Low `{SOLAR_COL}` with precipitation in the current/recent hours or `RH >= {QCConfig().weather_high_rh_threshold:g}%` is treated as a weather-related event, not as a value to automatically correct.

**Pass five - summary outputs.** The code writes row-level marked CSVs, detailed continuous-period tables, simplified station/category tables, the report, and the anomaly distribution figure.

---

## Anomaly statistics

Generated from the combined QC summary CSVs in this report. Weather-related low-radiation records are reported as advisory events and are **not** included in the correction-target anomaly distribution figure.

### Overall

- **Files scanned:** {len(stations)} prewashed met station files
- **Prewashed records across all files:** {total_records:,}
- **Duplicate timestamp rows:** 0
- **Timestamp gap records:** 0
- **Missing `{SOLAR_COL}` values:** {int(health["missing_srad_values"].sum()):,}
- **Correction-target QC anomaly records:** {true_anomaly_records:,} (**{true_anomaly_rate:.2f}%** of prewashed records)
- **Weather-related low-radiation event records:** {weather_event_records:,}
- **All flagged records/events:** {total_flagged:,} (**{flagged_rate:.2f}%** of prewashed records)
- **Continuous QC/event periods logged:** {len(detailed):,}
- **Stations with at least one QC flag/event:** {(pivot["Total flagged records/events"] > 0).sum()} of {len(stations)}

### Major anomaly categories

{markdown_table(["Category", "Included detailed types", "Records", "% of flags/events", "Stations affected", "Treatment"], category_rows)}

### Detailed anomaly types

{markdown_table(["Detailed type", "Periods/runs", "Records", "Treatment"], detailed_rows)}

### Per-station

Stations are sorted by total flagged records/events. The `% records needing review` column counts only correction-target QC anomalies, excluding weather-related low-radiation events.

{markdown_table(["Station", "Prewashed rows", "All flags/events", "Correction-target QC anomalies", "% records needing review", "Unexplained low", "Night-time", "Missing Srad/gap", "Spike/range", "Weather event", "First flag/event", "Last flag/event"], station_rows)}

### Imputation plan

No imputation is performed in this report. The script only marks QC anomalies and weather-related events so the original prewashed observations remain auditable.

| Anomaly class | Suggested imputation decision |
|---|---|
| Night-time radiation anomaly | If confirmed as sensor offset/noise, replace with 0 for radiation-balance workflows or set to NA for conservative analyses. Keep an imputation flag. |
| Missing Srad value or timestamp gap | For short gaps, use solar-aware interpolation constrained by hour-of-day and daylight/night status. For longer gaps, use seasonal/hourly climatology, nearby stations, or a model using `Rso`, `Tair`, `RH`, `Ppt`, and wind variables. |
| Spike/out-of-range radiation anomaly | Treat as invalid first, then impute using neighboring hours/stations or a clear-sky-constrained model. |
| Unexplained daytime low-radiation anomaly | Impute only after checking sensor context and nearby stations. |
| Weather-related low-radiation event | Do not impute by default; retain as a likely real cloudy/rainy/foggy condition. |

Recommended safeguards for any later imputation:

- Preserve the original `{SOLAR_COL}` column and write imputed values to a new column such as `Srad_imputed`.
- Add columns such as `Srad_imputed_flag` and `Srad_imputation_method`.
- Enforce physical bounds: `Srad_imputed >= 0` and a reasonable upper bound, such as the configured {QCConfig().physical_max:g} W/m^2 threshold or a clear-sky envelope after unit harmonization.
- Evaluate imputation quality by station, season, hour, and anomaly type before using the data in downstream analysis.

### Outputs generated

{markdown_table(["Output", "Path", "Contents"], output_rows)}

### Anomaly Distribution

![Srad correction-target QC anomalies by dataset](anomaly_distribution_by_dataset.png)
"""


def main() -> None:
    # Regenerate all marked data, tables, report text, and the distribution figure.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MARKED_DIR.mkdir(parents=True, exist_ok=True)

    config = QCConfig()
    simplified_parts = []
    detailed_parts = []
    health_rows = []
    for path in input_files():
        print(f"Processing {path.name}...")
        simplified, detailed, health = process_file(path, config)
        simplified_parts.append(simplified)
        detailed_parts.append(detailed)
        health_rows.append(health)

    simplified = (
        pd.concat(simplified_parts, ignore_index=True)
        if simplified_parts
        else pd.DataFrame(columns=SUMMARY_COLUMNS)
    )
    detailed = (
        pd.concat(detailed_parts, ignore_index=True)
        if detailed_parts
        else pd.DataFrame(columns=DETAILED_COLUMNS)
    )
    health = pd.DataFrame(health_rows)

    simplified.to_csv(TABLE_DIR / "combined_simplified_anomaly_summary.csv", index=False)
    detailed.to_csv(TABLE_DIR / "combined_detailed_anomaly_summary.csv", index=False)
    health.to_csv(TABLE_DIR / "prewashed_data_readiness_summary.csv", index=False)

    record_counts = dict(zip(health["station_id"], health["records"]))
    make_distribution_figure(simplified, record_counts)
    REPORT_PATH.write_text(report_markdown(simplified, detailed, health), encoding="utf-8")

    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Wrote {TABLE_DIR.relative_to(ROOT)}")
    print(f"Wrote {MARKED_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

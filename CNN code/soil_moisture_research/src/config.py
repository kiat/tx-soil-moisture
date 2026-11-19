"""Central configuration: paths, stations, splits, and feature groups.

All paths are resolved relative to the project root (the ``soil_moisture_research``
directory) so the project runs unchanged from a clone, a different checkout
location, or Google Colab. Override the data location with the environment
variable ``TXSON_PARQUET`` if the dataset lives elsewhere.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_FIG_DIR = PROJECT_ROOT / "report" / "figures"

PARQUET_NAME = "TxSON33_modeling_complete.parquet"

# Candidate locations searched (first hit wins). TXSON_PARQUET overrides all.
_PARQUET_CANDIDATES = [
    os.environ.get("TXSON_PARQUET", ""),
    str(DATA_DIR / PARQUET_NAME),
    # local developer checkout used during this project
    str(Path.home() / "tx-soil-moisture" / "tx-soil-moisture" / "datasets"
        / "txson33_complete_for_modeling" / PARQUET_NAME),
    # Google Colab + Drive
    "/content/drive/MyDrive/TxSON33_CNN_results/" + PARQUET_NAME,
    "/content/" + PARQUET_NAME,
]

# Public GitHub mirror (verified working path in the kiat/tx-soil-moisture repo).
DATA_URL = ("https://raw.githubusercontent.com/kiat/tx-soil-moisture/main/"
            "datasets/txson33_complete_for_modeling/" + PARQUET_NAME)


def find_parquet() -> str:
    """Return the first existing dataset path, or the default download target."""
    for c in _PARQUET_CANDIDATES:
        if c and os.path.exists(c):
            return c
    return str(DATA_DIR / PARQUET_NAME)


# ---------------------------------------------------------------------------
# Target, stations, splits
# ---------------------------------------------------------------------------
TARGET = "SWC_20"                    # soil water content at 20 cm (m^3/m^3)

ALL_YEARS = list(range(2015, 2025))  # 2014 / 2025 are partial years, excluded
VAL_YEAR = 2019                      # fixed validation year (never a test year)

# Four TxSON stations sharing 10 complete calendar years.
SELECTED_STATIONS = ["FD11", "FD02", "CB06", "CB25"]

# The five held-out (station, year) test cases used in the multi-station study.
TEST_STATION_YEARS = [
    ("FD11", 2023), ("FD11", 2020),
    ("CB06", 2023), ("CB25", 2020), ("FD02", 2020),
]

SEED = 42

# ---------------------------------------------------------------------------
# Feature configurations (A-F) -- exactly those evaluated in the Nov-17 study.
# Each list is the *exogenous* part; SWC_20 (history) is always prepended.
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "swc":             [],                                   # A: SWC_20 only (baseline)
    "existing_rain":   ["Ppt_log"],                          # B
    "lagged_rain":     ["Ppt_log", "Ppt_log_lag1",
                        "Ppt_log_lag3", "Ppt_log_lag6"],     # C
    "cumulative_rain": ["Ppt_6h", "Ppt_24h", "Ppt_72h"],     # D
    "temperature":     ["Tair"],                             # E (air temperature)
    "humidity":        ["RH"],
    "seasonality":     ["sin_doy", "cos_doy"],
    "wind":            ["Wind_u", "Wind_v"],
    "soil_temp":       ["T_20"],                             # genuine soil temp @20cm
    "combined_met":    ["Ppt_log", "Tair", "RH",
                        "sin_doy", "cos_doy"],               # F
}

# Human-readable labels used in tables/figures and matching the saved result CSVs.
CONFIG_LABELS = {
    "swc":             "A: SWC_20 only",
    "existing_rain":   "B: + existing rain",
    "lagged_rain":     "C: + lagged rain",
    "cumulative_rain": "D: + cumulative rain",
    "temperature":     "E: + air temperature",
    "combined_met":    "F: combined met",
}


def feature_columns(group: str) -> list[str]:
    """Full input column list for a feature group (TARGET always first)."""
    if group not in FEATURE_GROUPS:
        raise KeyError(f"unknown feature group {group!r}; choices: {list(FEATURE_GROUPS)}")
    return [TARGET] + [c for c in FEATURE_GROUPS[group] if c != TARGET]

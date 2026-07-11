"""Central configuration for parameter families and modeling metadata."""
from __future__ import annotations

from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# Parameter families
# ---------------------------------------------------------------------------
SOIL_MOISTURE = "soil_moisture"
SOIL_TEMPERATURE = "soil_temperature"
AIR_TEMPERATURE = "air_temperature"
RELATIVE_HUMIDITY = "relative_humidity"
SOLAR_RADIATION = "solar_radiation"
WIND_SPEED = "wind_speed"
WIND_DIRECTION = "wind_direction"
PRECIPITATION = "precipitation"

PARAM_FAMILY: Dict[str, str] = {
    # Soil moisture
    "SWC_5": SOIL_MOISTURE,
    "SWC_10": SOIL_MOISTURE,
    "SWC_20": SOIL_MOISTURE,
    "SWC_50": SOIL_MOISTURE,
    # Soil temperature
    "T_5": SOIL_TEMPERATURE,
    "T_10": SOIL_TEMPERATURE,
    "T_20": SOIL_TEMPERATURE,
    "T_50": SOIL_TEMPERATURE,
    # MET variables
    "Tair": AIR_TEMPERATURE,
    "RH": RELATIVE_HUMIDITY,
    "Srad": SOLAR_RADIATION,
    "Wind speed": WIND_SPEED,
    "Wind direction": WIND_DIRECTION,
    "Ppt": PRECIPITATION,
}

ALL_SOIL_PARAMS = [p for p, fam in PARAM_FAMILY.items() if fam in {SOIL_MOISTURE, SOIL_TEMPERATURE}]
ALL_MET_PARAMS = [p for p, fam in PARAM_FAMILY.items() if fam not in {SOIL_MOISTURE, SOIL_TEMPERATURE}]

# ---------------------------------------------------------------------------
# Short-gap interpolation hints (per family)
# ---------------------------------------------------------------------------
# Supported interpolation tags: "pchip", "time", "linear", "zero", "wind_angle"
SHORT_INTERP_BY_FAMILY = {
    SOIL_MOISTURE: "pchip",
    SOIL_TEMPERATURE: "time",
    AIR_TEMPERATURE: "time",
    RELATIVE_HUMIDITY: "linear",
    SOLAR_RADIATION: "linear",
    WIND_SPEED: "linear",
    WIND_DIRECTION: "wind_angle",
    PRECIPITATION: "zero",
}

# ---------------------------------------------------------------------------
# Medium-gap exogenous inputs (per family)
# ---------------------------------------------------------------------------
FAMILY_EXOG_MAP = {
    SOIL_MOISTURE: ["Ppt"],
    SOIL_TEMPERATURE: ["Tair", "Srad"],
    AIR_TEMPERATURE: ["Srad"],
    RELATIVE_HUMIDITY: ["Tair", "Ppt"],
    SOLAR_RADIATION: ["Tair"],
    WIND_SPEED: ["Srad", "Ppt", "Tair"],
    WIND_DIRECTION: ["Wind speed", "Tair"],
    PRECIPITATION: [],
}

# ---------------------------------------------------------------------------
# Long-gap driver dependencies
# ---------------------------------------------------------------------------
FAMILY_DRIVER_DEPENDENCIES = {
    SOIL_MOISTURE: ["Ppt", "Tair"],
    SOIL_TEMPERATURE: ["Ppt", "Tair", "Srad"],
    AIR_TEMPERATURE: ["Srad"],
    RELATIVE_HUMIDITY: ["Tair", "Ppt"],
    SOLAR_RADIATION: ["Tair"],
    WIND_SPEED: ["Ppt", "Tair", "Srad"],
    WIND_DIRECTION: ["Wind speed", "Tair"],
    PRECIPITATION: [],
}

# ---------------------------------------------------------------------------
# Recommended fill order (honors dependencies)
# ---------------------------------------------------------------------------
FILL_ORDER: Sequence[str] = [
    "Tair",
    "Srad",
    "RH",
    "Wind speed",
    "Wind direction",
    "Ppt",
    # After MET drivers, fall back to soil moisture/temperature
    "SWC_5", "SWC_10", "SWC_20", "SWC_50",
    "T_5", "T_10", "T_20", "T_50",
]

ALL_PARAMS = list(FILL_ORDER)


def get_family(param: str) -> str:
    return PARAM_FAMILY.get(param, SOIL_MOISTURE)


def short_interp_for(param: str) -> str:
    return SHORT_INTERP_BY_FAMILY.get(get_family(param), "linear")


def exog_for(param: str) -> List[str]:
    return FAMILY_EXOG_MAP.get(get_family(param), [])


def driver_dependencies(param: str) -> List[str]:
    return FAMILY_DRIVER_DEPENDENCIES.get(get_family(param), [])

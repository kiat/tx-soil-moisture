"""Apply confirmed sensor-level QC masks to staged output files.

The script writes a new non-destructive stage:

    output/Station{site}_filled_sensor_qc.csv

By default it masks only rows classified as bad_sensor_candidate by
sensor_qc_decisions.py. Other review categories remain unchanged until they are
manually confirmed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "sensor_qc_reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Apply sensor QC masks to station outputs.")
    parser.add_argument("--station", type=str, nargs="*", help="Station IDs/site codes to process.")
    parser.add_argument("--write", action="store_true", help="Write *_filled_sensor_qc.csv files.")
    parser.add_argument(
        "--mask-localized-bound-values",
        action="store_true",
        help="Also mask exact-bound values for localized_bound_values_review rows.",
    )
    return parser.parse_args()


def discover_stations() -> List[str]:
    pat = re.compile(r"Station(.+)_filled_verylonggaps_repaired\.csv")
    return sorted(
        m.group(1)
        for path in OUT_DIR.glob("Station*_filled_verylonggaps_repaired.csv")
        if (m := pat.match(path.name))
    )


def latest_path_for(station: str) -> Path:
    candidates = [
        OUT_DIR / f"Station{station}_filled_verylonggaps_repaired.csv",
        OUT_DIR / f"Station{station}_filled_verylonggaps.csv",
        OUT_DIR / f"Station{station}_filled_longgaps_repaired.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def read_station(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    return df.sort_index()


def load_decisions() -> pd.DataFrame:
    path = REPORT_DIR / "sensor_qc_decisions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run sensor_qc_decisions.py first.")
    return pd.read_csv(path)


def mask_for_row(df: pd.DataFrame, row: pd.Series, mask_localized_bound_values: bool) -> pd.Series:
    param = row["Parameter"]
    decision = row["QC Decision"]
    if param not in df.columns:
        return pd.Series(False, index=df.index)

    if decision == "bad_sensor_candidate":
        return df[param].notna()

    if decision == "localized_bound_values_review" and mask_localized_bound_values:
        values = pd.to_numeric(df[param], errors="coerce")
        if param.startswith("SWC_"):
            return values.eq(0.0) | values.eq(0.6)
        if param.startswith("T_"):
            return values.eq(-30.0) | values.eq(60.0)

    return pd.Series(False, index=df.index)


def main() -> None:
    args = parse_args()
    stations = args.station if args.station else discover_stations()
    decisions = load_decisions()
    decisions["Station"] = decisions["Station"].astype(str)
    selected = decisions[decisions["Station"].isin(stations)]

    detail_rows: List[Dict[str, object]] = []
    station_rows: List[Dict[str, object]] = []

    for station in stations:
        input_path = latest_path_for(station)
        if not input_path.exists():
            continue
        df = read_station(input_path)
        station_masked = 0

        for _, row in selected[selected["Station"] == station].iterrows():
            param = row["Parameter"]
            if param not in df.columns:
                continue
            mask = mask_for_row(df, row, args.mask_localized_bound_values)
            masked_hours = int(mask.sum())
            if masked_hours:
                station_masked += masked_hours
                for ts in df.index[mask]:
                    detail_rows.append(
                        {
                            "Station": station,
                            "Parameter": param,
                            "Timestamp": ts,
                            "QC Decision": row["QC Decision"],
                            "Recommended Action": row["Recommended Action"],
                        }
                    )
                df.loc[mask, param] = pd.NA

        station_rows.append(
            {
                "Station": station,
                "Input File": input_path.name,
                "Output File": f"Station{station}_filled_sensor_qc.csv",
                "Newly Masked Hours": station_masked,
            }
        )

        if args.write:
            df.to_csv(OUT_DIR / f"Station{station}_filled_sensor_qc.csv", na_rep="NaN")

    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(station_rows).to_csv(REPORT_DIR / "sensor_qc_mask_station_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(REPORT_DIR / "sensor_qc_masked_points.csv", index=False)

    print("Sensor QC mask step complete.")
    print(f"Stations processed: {len(station_rows)}")
    print(f"Newly masked hours: {sum(row['Newly Masked Hours'] for row in station_rows)}")
    print(f"Reports written under: {REPORT_DIR}")
    if args.write:
        print("Station files written to output/*_filled_sensor_qc.csv")
    else:
        print("Dry run only. Add --write to write station files.")


if __name__ == "__main__":
    main()

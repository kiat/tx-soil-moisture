"""Apply manually reviewed QC masks after automatic sensor QC.

This non-destructive stage reads:

    output/Station{site}_filled_sensor_qc.csv

and writes, only for affected stations:

    output/Station{site}_filled_manual_qc.csv

The mask definitions live in manual_qc_masks.csv so manual decisions are
auditable and reproducible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "manual_qc_reports"
DEFAULT_MASK_FILE = BASE_DIR / "manual_qc_masks.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Apply manual QC masks to sensor-QC outputs.")
    parser.add_argument("--mask-file", type=Path, default=DEFAULT_MASK_FILE)
    parser.add_argument("--station", type=str, nargs="*", help="Optional station/site codes to process.")
    parser.add_argument("--write", action="store_true", help="Write *_filled_manual_qc.csv files.")
    return parser.parse_args()


def read_station(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    return df.sort_index()


def input_path_for(station: str) -> Path:
    candidates = [
        OUT_DIR / f"Station{station}_filled_sensor_qc.csv",
        OUT_DIR / f"Station{station}_filled_verylonggaps_repaired.csv",
        OUT_DIR / f"Station{station}_filled_verylonggaps.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def load_masks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manual QC mask file not found: {path}")
    masks = pd.read_csv(path, parse_dates=["Start", "End"])
    required = {"Station", "Parameter", "Start", "End", "Decision", "Reason"}
    missing = required - set(masks.columns)
    if missing:
        raise ValueError(f"Manual QC mask file is missing columns: {sorted(missing)}")
    masks["Station"] = masks["Station"].astype(str)
    return masks[masks["Decision"].eq("mask_and_refill")].copy()


def main() -> None:
    args = parse_args()
    masks = load_masks(args.mask_file)
    if args.station:
        masks = masks[masks["Station"].isin(args.station)]

    stations = sorted(masks["Station"].unique())
    detail_rows: List[Dict[str, object]] = []
    station_rows: List[Dict[str, object]] = []

    for station in stations:
        input_path = input_path_for(station)
        if not input_path.exists():
            station_rows.append(
                {
                    "Station": station,
                    "Input File": input_path.name,
                    "Output File": f"Station{station}_filled_manual_qc.csv",
                    "Masked Hours": 0,
                    "Status": "missing_input",
                }
            )
            continue

        df = read_station(input_path)
        station_masked = 0
        station_masks = masks[masks["Station"] == station]

        for _, row in station_masks.iterrows():
            param = row["Parameter"]
            if param not in df.columns:
                detail_rows.append(
                    {
                        "Station": station,
                        "Parameter": param,
                        "Start": row["Start"],
                        "End": row["End"],
                        "Masked Hours": 0,
                        "Refill Method": row.get("Refill Method", "auto"),
                        "Reason": row["Reason"],
                        "Notes": row.get("Notes", ""),
                        "Status": "missing_parameter",
                    }
                )
                continue

            start = pd.Timestamp(row["Start"])
            end = pd.Timestamp(row["End"])
            mask = (df.index >= start) & (df.index <= end) & df[param].notna()
            masked_hours = int(mask.sum())
            if masked_hours:
                df.loc[mask, param] = pd.NA
                station_masked += masked_hours

            detail_rows.append(
                {
                    "Station": station,
                    "Parameter": param,
                    "Start": start,
                    "End": end,
                    "Masked Hours": masked_hours,
                    "Refill Method": row.get("Refill Method", "auto"),
                    "Reason": row["Reason"],
                    "Notes": row.get("Notes", ""),
                    "Status": "masked" if masked_hours else "no_nonmissing_values",
                }
            )

        output_path = OUT_DIR / f"Station{station}_filled_manual_qc.csv"
        station_rows.append(
            {
                "Station": station,
                "Input File": input_path.name,
                "Output File": output_path.name,
                "Masked Hours": station_masked,
                "Status": "written" if args.write else "dry_run",
            }
        )
        if args.write:
            df.to_csv(output_path, na_rep="NaN")

    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(station_rows).to_csv(REPORT_DIR / "manual_qc_mask_station_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(REPORT_DIR / "manual_qc_mask_detail.csv", index=False)

    total_masked = sum(row["Masked Hours"] for row in station_rows)
    print("Manual QC mask step complete.")
    print(f"Stations processed: {len(station_rows)}")
    print(f"Masked hours: {total_masked}")
    print(f"Reports written under: {REPORT_DIR}")
    if args.write:
        print("Station files written to output/*_filled_manual_qc.csv")
    else:
        print("Dry run only. Add --write to write station files.")


if __name__ == "__main__":
    main()

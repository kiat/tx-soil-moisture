#!/usr/bin/env python3
"""Orchestrate the TxSON 33-station soil imputation workflow.

This runner is the preferred entry point for the current 33-station workflow.
It keeps the individual scripts available for debugging, but gives users one
clear command for the normal run order.

Examples:
    python imputation_pipeline.py --stage all
    python imputation_pipeline.py --stage qc
    python imputation_pipeline.py --stage final
    python imputation_pipeline.py --stage all --dry-run
    python imputation_pipeline.py --stage all --station CB01 FD08
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / ".." / ".." / "datasets" / "TxSON_data_2026-02-24"


@dataclass(frozen=True)
class Step:
    name: str
    command: List[str]


STAGE_ORDER = [
    "clean",
    "short",
    "medium",
    "validate-medium",
    "long",
    "validate-long",
    "verylong",
    "validate-verylong",
    "qc-before-sensor",
    "sensor-decisions",
    "sensor-mask",
    "qc-after-sensor",
    "final",
    "qc-final",
]

STAGE_GROUPS = {
    "all": STAGE_ORDER,
    "soil": STAGE_ORDER,
    "clean": ["clean"],
    "short": ["short"],
    "medium": ["medium", "validate-medium"],
    "long": ["long", "validate-long"],
    "verylong": ["verylong", "validate-verylong"],
    "qc": ["qc-before-sensor", "sensor-decisions", "sensor-mask", "qc-after-sensor"],
    "final": ["final", "qc-final"],
}

STALE_PATTERNS_BY_STAGE = {
    "clean": [
        "cleaned_data/Station*_cleaned_data.csv",
        "missing_data/Station*_missing_data.csv",
        "raw_merged_data/raw_merged_station_*.csv",
        "stage0_summary.csv",
        "shortgaps_summary.csv",
    ],
    "short": [
        "output/Station*_filled_shortgaps.csv",
        "output/Station*_shortgap_fill_detail.csv",
    ],
    "medium": [
        "output/Station*_filled_mediumgaps.csv",
        "output/Station*_mediumgap_fill_detail.csv",
        "output/Station*_filled_mediumgaps_repaired.csv",
        "output/Station*_mediumgap_fill_detail_repaired.csv",
        "mediumgaps_validation_summary.csv",
        "mediumgaps_rejected_segments.csv",
        "mediumgaps_validation_station_summary.csv",
    ],
    "long": [
        "output/Station*_filled_longgaps.csv",
        "output/Station*_longgap_fill_detail.csv",
        "output/Station*_filled_longgaps_repaired.csv",
        "output/Station*_longgap_fill_detail_repaired.csv",
        "longgaps_validation_summary.csv",
        "longgaps_rejected_segments.csv",
        "longgaps_validation_station_summary.csv",
    ],
    "verylong": [
        "output/Station*_filled_verylonggaps.csv",
        "output/Station*_verylonggap_fill_detail.csv",
        "output/Station*_filled_verylonggaps_repaired.csv",
        "output/Station*_verylonggap_fill_detail_repaired.csv",
        "verylonggaps_validation_summary.csv",
        "verylonggaps_review_segments.csv",
        "verylonggaps_repaired_points.csv",
        "verylonggaps_validation_station_summary.csv",
    ],
    "sensor-mask": [
        "output/Station*_filled_sensor_qc.csv",
        "sensor_qc_reports",
    ],
    "final": [
        "output/Station*_filled_final.csv",
        "output/Station*_final_residual_fill_detail.csv",
        "final_qc_reports",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run the TxSON 33-station soil imputation workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_GROUPS),
        default="all",
        help="Workflow group to run.",
    )
    parser.add_argument("--station", type=str, nargs="*", help="Optional station/site codes, e.g. CB01 FD08.")
    parser.add_argument("--param", type=str, nargs="*", help="Optional soil parameters for fill/validation scripts.")
    parser.add_argument("--soil-base-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing soil .dat files.")
    parser.add_argument("--met-base-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing MET .dat files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and cleanup actions without running.")
    parser.add_argument(
        "--no-clean-stale",
        action="store_true",
        help="Do not delete stale downstream generated outputs before running.",
    )
    return parser.parse_args()


def discover_stations(soil_base_dir: Path) -> List[str]:
    if not soil_base_dir.exists():
        raise FileNotFoundError(f"Soil data directory does not exist: {soil_base_dir}")
    stations = []
    for path in sorted(soil_base_dir.glob("*.dat")):
        name = path.stem
        if name.endswith("_met") or name.startswith("MET_"):
            continue
        if name.startswith("SM_"):
            stations.append(name.removeprefix("SM_"))
        else:
            stations.append(name)
    if not stations:
        raise FileNotFoundError(f"No soil station .dat files found in {soil_base_dir}")
    return sorted(dict.fromkeys(stations))


def selected_stages(stage_group: str) -> List[str]:
    return list(STAGE_GROUPS[stage_group])


def command_with_selection(base: Sequence[str], stations: Sequence[str] | None, params: Sequence[str] | None) -> List[str]:
    command = list(base)
    if stations:
        command.extend(["--station", *stations])
    if params:
        command.extend(["--param", *params])
    return command


def build_steps(args: argparse.Namespace, stages: Sequence[str], stations: Sequence[str]) -> List[Step]:
    py = sys.executable
    steps: List[Step] = []

    if "clean" in stages:
        for station in stations:
            steps.append(
                Step(
                    f"clean:{station}",
                    [
                        py,
                        "datacleaning.py",
                        "--station",
                        station,
                        "--soil-base-dir",
                        str(args.soil_base_dir),
                        "--met-base-dir",
                        str(args.met_base_dir),
                    ],
                )
            )

    stations_arg = args.station
    params_arg = args.param
    stage_commands = {
        "short": command_with_selection([py, "Shortgaps.py"], stations_arg, params_arg),
        "medium": command_with_selection([py, "Mediumgaps.py"], stations_arg, params_arg),
        "validate-medium": command_with_selection([py, "validate_mediumgaps.py", "--write-repaired"], stations_arg, params_arg),
        "long": command_with_selection([py, "Longgaps.py"], stations_arg, params_arg),
        "validate-long": command_with_selection([py, "validate_longgaps.py", "--write-repaired"], stations_arg, params_arg),
        "verylong": command_with_selection([py, "VeryLongGaps.py"], stations_arg, params_arg),
        "validate-verylong": command_with_selection([py, "validate_verylonggaps.py", "--write-repaired"], stations_arg, params_arg),
        "qc-before-sensor": command_with_selection([py, "final_qc_summary.py"], stations_arg, params_arg),
        "sensor-decisions": [py, "sensor_qc_decisions.py"],
        "sensor-mask": [py, "apply_sensor_qc_masks.py", "--write", *(["--station", *stations_arg] if stations_arg else [])],
        "qc-after-sensor": command_with_selection([py, "final_qc_summary.py"], stations_arg, params_arg),
        "final": command_with_selection([py, "FinalResidualGaps.py"], stations_arg, params_arg),
        "qc-final": command_with_selection([py, "final_qc_summary.py"], stations_arg, params_arg),
    }

    for stage in stages:
        if stage == "clean":
            continue
        steps.append(Step(stage, stage_commands[stage]))
    return steps


def cleanup_start_index(stages: Sequence[str]) -> int | None:
    order_index = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    indexes = [order_index[stage] for stage in stages if stage in order_index]
    return min(indexes) if indexes else None


def stale_patterns_for_run(stages: Sequence[str]) -> List[str]:
    start = cleanup_start_index(stages)
    if start is None:
        return []
    selected = set(STAGE_ORDER[start:])
    patterns: List[str] = []
    for stage, stage_patterns in STALE_PATTERNS_BY_STAGE.items():
        if stage in selected:
            patterns.extend(stage_patterns)
    return patterns


def station_scoped_path(path: Path, stations: Sequence[str] | None) -> bool:
    if not stations:
        return True
    name = path.name
    if name.startswith("Station"):
        return any(name.startswith(f"Station{station}_") for station in stations)
    if name.startswith("raw_merged_station_"):
        return any(name.startswith(f"raw_merged_station_{station}") for station in stations)
    return True


def clean_stale_outputs(patterns: Iterable[str], dry_run: bool, stations: Sequence[str] | None = None) -> None:
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        for path in matches:
            if path in seen:
                continue
            if not station_scoped_path(path, stations):
                continue
            seen.add(path)
            rel = path.relative_to(ROOT)
            print(f"cleanup: {rel}")
            if dry_run:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def run_step(step: Step, dry_run: bool) -> None:
    print(f"\n>>> [{step.name}] {' '.join(step.command)}")
    if dry_run:
        return
    subprocess.run(step.command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    args.soil_base_dir = args.soil_base_dir.expanduser().resolve()
    args.met_base_dir = args.met_base_dir.expanduser().resolve()

    stages = selected_stages(args.stage)
    stations = args.station if args.station else discover_stations(args.soil_base_dir)
    steps = build_steps(args, stages, stations)

    print("TxSON 33-station imputation runner")
    print(f"Stage group: {args.stage}")
    print(f"Stages: {', '.join(stages)}")
    print(f"Stations: {', '.join(stations)}")
    if args.param:
        print(f"Parameters: {', '.join(args.param)}")

    if not args.no_clean_stale:
        patterns = stale_patterns_for_run(stages)
        if patterns:
            print("\nRemoving stale generated outputs for selected stage range...")
            clean_stale_outputs(patterns, args.dry_run, args.station)
    else:
        print("\nSkipping stale-output cleanup because --no-clean-stale was supplied.")

    for step in steps:
        run_step(step, args.dry_run)

    print("\nPipeline runner finished successfully.")


if __name__ == "__main__":
    main()

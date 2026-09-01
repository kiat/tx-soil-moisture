#!/usr/bin/env python3
"""Orchestrate the TxSON soil workflow and isolated MET stages.

This runner is the preferred entry point for the current 33-station workflow.
It keeps the individual scripts available for debugging, but gives users one
clear command for the normal run order.

Examples:
    python imputation_pipeline.py --stage all
    python imputation_pipeline.py --stage qc
    python imputation_pipeline.py --stage final
    python imputation_pipeline.py --stage all --dry-run
    python imputation_pipeline.py --stage all --station CB01 FD08
    python imputation_pipeline.py --stage met --station FD02
    python imputation_pipeline.py --stage met-full --station FD02
    python imputation_pipeline.py --stage met-ppt --station FD02
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from param_config import ALL_MET_PARAMS, ALL_SOIL_PARAMS


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
    "manual-mask",
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
    "qc": ["qc-before-sensor", "sensor-decisions", "sensor-mask", "manual-mask", "qc-after-sensor"],
    "final": ["final", "qc-final"],
    "met": ["met"],
    "met-full": ["met-full"],
    "met-ppt": ["met-ppt"],
}

STALE_PATTERNS_BY_STAGE = {
    "clean": [
        "cleaned_data/Station*_cleaned_data.csv",
        "missing_data/Station*_missing_data.csv",
        "raw_merged_data/raw_merged_station_*.csv",
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
        "manual_qc_reports",
        "output/Station*_filled_manual_qc.csv",
    ],
    "final": [
        "output/Station*_filled_final.csv",
        "output/Station*_final_residual_fill_detail.csv",
        "final_qc_reports",
    ],
    "met": [
        "met_output/Station*_met_filled_shortgaps.csv",
        "met_qc_reports/met_station_parameter_summary.csv",
        "met_qc_reports/met_gap_inventory.csv",
        "met_qc_reports/ppt_source_comparison_summary.csv",
    ],
    "met-full": [
        "met_output/Station*_met_filled_allgaps.csv",
        "met_qc_reports/model_fill",
    ],
    "met-ppt": [
        "met_output/Station*_met_filled_complete.csv",
        "met_qc_reports/ppt_model_fill",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run the TxSON soil workflow or an isolated MET stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_GROUPS),
        default="all",
        help="Workflow group to run.",
    )
    parser.add_argument("--station", type=str, nargs="*", help="Optional station/site codes, e.g. CB01 FD08.")
    parser.add_argument("--param", type=str, nargs="*", help="Optional parameters for the selected soil or MET stage.")
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
    targeted = bool(stations_arg)

    def final_qc_command(report_name: str, input_stage: str) -> List[str]:
        base = [py, "final_qc_summary.py", "--input-stage", input_stage]
        if targeted:
            base.extend(["--report-dir", str(ROOT / "targeted_qc_reports" / report_name)])
        return command_with_selection(base, stations_arg, params_arg)

    sensor_report_dir = ROOT / "targeted_qc_reports" / "sensor_qc"
    manual_report_dir = ROOT / "targeted_qc_reports" / "manual_qc"
    before_sensor_dir = ROOT / "targeted_qc_reports" / "before_sensor"

    def validation_command(script: str, report_name: str) -> List[str]:
        base = [py, script, "--write-repaired"]
        if targeted:
            base.extend(["--report-dir", str(ROOT / "targeted_qc_reports" / report_name)])
        return command_with_selection(base, stations_arg, params_arg)

    def met_command(*mode_flags: str) -> List[str]:
        return command_with_selection(
            [
                py,
                "MetGaps.py",
                *mode_flags,
                "--write",
                "--soil-base-dir",
                str(args.soil_base_dir),
                "--met-base-dir",
                str(args.met_base_dir),
            ],
            stations_arg,
            params_arg,
        )

    stage_commands = {
        "short": command_with_selection([py, "Shortgaps.py"], stations_arg, params_arg),
        "medium": command_with_selection([py, "Mediumgaps.py"], stations_arg, params_arg),
        "validate-medium": validation_command("validate_mediumgaps.py", "medium_validation"),
        "long": command_with_selection([py, "Longgaps.py"], stations_arg, params_arg),
        "validate-long": validation_command("validate_longgaps.py", "long_validation"),
        "verylong": command_with_selection([py, "VeryLongGaps.py"], stations_arg, params_arg),
        "validate-verylong": validation_command("validate_verylonggaps.py", "verylong_validation"),
        "qc-before-sensor": final_qc_command("before_sensor", "verylong-repaired"),
        "sensor-decisions": [
            py,
            "sensor_qc_decisions.py",
            *(["--input-dir", str(before_sensor_dir), "--report-dir", str(sensor_report_dir)] if targeted else []),
        ],
        "sensor-mask": [
            py,
            "apply_sensor_qc_masks.py",
            "--write",
            *(["--decision-dir", str(sensor_report_dir), "--report-dir", str(sensor_report_dir)] if targeted else []),
            *(["--station", *stations_arg] if stations_arg else []),
        ],
        "manual-mask": [
            py,
            "apply_manual_qc_masks.py",
            "--write",
            *(["--report-dir", str(manual_report_dir)] if targeted else []),
            *(["--station", *stations_arg] if stations_arg else []),
        ],
        "qc-after-sensor": final_qc_command("after_sensor", "post-qc"),
        "final": command_with_selection([py, "FinalResidualGaps.py"], stations_arg, params_arg),
        "qc-final": final_qc_command("final", "final"),
        "met": met_command(),
        "met-full": met_command("--full", "--repair-review"),
        "met-ppt": met_command("--ppt-full"),
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
        return [
            pattern
            for stage in stages
            for pattern in STALE_PATTERNS_BY_STAGE.get(stage, [])
        ]
    selected = set(STAGE_ORDER[start:])
    patterns: List[str] = []
    for stage, stage_patterns in STALE_PATTERNS_BY_STAGE.items():
        if stage in selected:
            patterns.extend(stage_patterns)
    return patterns


def station_scoped_path(path: Path, stations: Sequence[str] | None) -> bool:
    if not stations:
        return True
    if path.is_dir():
        # A selected-station run must not delete full-batch global reports.
        return False
    name = path.name
    if name.startswith("Station"):
        return any(name.startswith(f"Station{station}_") for station in stations)
    if name.startswith("raw_merged_station_"):
        return any(name.startswith(f"raw_merged_station_{station}") for station in stations)
    return False


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

    stages = list(STAGE_GROUPS[args.stage])
    if args.param:
        if args.stage == "met-ppt":
            allowed = {"Ppt"}
        elif args.stage in {"met", "met-full"}:
            allowed = set(ALL_MET_PARAMS)
        else:
            allowed = set(ALL_SOIL_PARAMS)
        unsupported = sorted(set(args.param) - allowed)
        if unsupported:
            scope = (
                "MET"
                if args.stage in {"met", "met-full", "met-ppt"}
                else "soil"
            )
            raise ValueError(f"Unsupported {scope} parameter(s) for stage {args.stage}: {', '.join(unsupported)}")
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
        if args.param and args.stage in {"met", "met-full"}:
            # Parameter-scoped MET reruns merge into existing station files.
            # Keep those files available so unselected columns are preserved.
            patterns = [
                pattern
                for pattern in patterns
                if not pattern.startswith("met_output/")
            ]
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

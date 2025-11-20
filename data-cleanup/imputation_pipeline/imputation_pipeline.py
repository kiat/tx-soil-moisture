#!/usr/bin/env python3
"""
imputation_pipeline.py – master runner for *data‑cleanup* branch
=======================================================
This script assumes **all utilities now live in the folder**

    data-cleanup/
        datacleaning.py            # step 0 – build cleaned CSV & gap logs
        Shortgaps.py               # step 1 – fill <24 h gaps
        Mediumgaps.py              # step 2 – fill 1–7 d gaps
        Longgaps.py                # step 3 – fill 7–30 d gaps
        VeryLongGaps.py            # step 4 – fill ≥30 d gaps

The workflow:
    0. `datacleaning.py --station N`   for N = 1…6
    1. `Shortgaps.py`   -> …_filled_shortgaps.csv
    2. `Mediumgaps.py`  -> …_filled_mediumgaps.csv
    3. `Longgaps.py`    -> …_filled_longgaps.csv
    4. `VeryLongGaps.py`-> …_filled_verylonggaps.csv



Usage examples
--------------
    # Run every station & every stage (default)
    python imputation_pipeline.py

    # Only Station 3 (all stages)
    python imputation_pipeline.py --station 3

    # Dry‑run: show exact commands without executing
    python imputation_pipeline.py --dry
"""

import subprocess, sys
from pathlib import Path
import argparse

ROOT       = Path(__file__).resolve().parent
CLEAN_DIR  = ROOT

# ---------- stage 0: data cleaning (creates cleaned CSV + gap logs) ----------
stage0 = [
    [sys.executable, "datacleaning.py", "--station", str(i)]
    for i in range(1, 7)               # Stations 1–6
]

# ---------- stages 1‑4: gap filling scripts (run in order) -------------------
stage_gaps = [
    [sys.executable, "Shortgaps.py"],
    [sys.executable, "Mediumgaps.py"],
    [sys.executable, "Longgaps.py"],
    [sys.executable, "VeryLongGaps.py"],
]

# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path, dry: bool):
    print(f"\n>>> {' '.join(cmd)}   (cwd={cwd.relative_to(ROOT)})")
    if dry:
        return
    exit_code = subprocess.call(cmd, cwd=cwd)
    if exit_code != 0:
        raise SystemExit(f"✗ step failed (exit code {exit_code}) – aborting.")


def main():
    ap = argparse.ArgumentParser("Run the full imputation pipeline (data‑cleanup branch)")
    ap.add_argument("--station", type=int, help="process only this station id (1‑6)")
    ap.add_argument("--dry", action="store_true", help="print commands without executing")
    args = ap.parse_args()

    # If user restricted to one station, rewrite stage0 commands accordingly
    if args.station:
        global stage0
        stage0 = [[sys.executable, "datacleaning.py", "--station", str(args.station)]]

    try:
        for cmd in stage0 + stage_gaps:
            run(cmd, cwd=CLEAN_DIR, dry=args.dry)
    except SystemExit as e:
        print(e)
        sys.exit(1)

    print("\n✓ Pipeline finished successfully!")


if __name__ == "__main__":
    main()

"""Create sensor-level QC decisions from final QC summaries.

This script does not modify station data. It translates the suspicious sensor
metrics from final_qc_summary.py into a small decision table that can be
reviewed before residual gap filling.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FINAL_QC_DIR = BASE_DIR / "final_qc_reports"
REPORT_DIR = BASE_DIR / "sensor_qc_reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build sensor-level QC decision tables.")
    parser.add_argument("--near-zero-bad", type=float, default=0.9)
    parser.add_argument("--near-zero-review", type=float, default=0.5)
    parser.add_argument("--min-hours", type=int, default=720)
    return parser.parse_args()


def decide(row: pd.Series, args: argparse.Namespace) -> Tuple[str, str, str]:
    flags = set(str(row.get("Flags", "")).split(";")) - {""}
    param = str(row["Parameter"])
    near_zero_fraction = float(row.get("SWC Near-Zero Fraction", 0.0))
    nonmissing = int(row.get("Nonmissing Hours", 0))

    has_near_zero = "swc_near_zero_dominant" in flags
    has_low_var = "swc_low_variability" in flags
    has_long_constant = "long_constant_run" in flags
    has_exact_bound = "exact_lower_bound_values" in flags or "exact_upper_bound_values" in flags

    if param.startswith("SWC_") and nonmissing >= args.min_hours:
        if near_zero_fraction >= args.near_zero_bad and (has_low_var or has_long_constant or has_exact_bound):
            return (
                "bad_sensor_candidate",
                "exclude_sensor_before_final_fill",
                "Most non-missing values are near zero and the column is low-variability or has long constant runs.",
            )
        if near_zero_fraction >= args.near_zero_review and (has_long_constant or has_exact_bound):
            return (
                "partial_or_bad_sensor_review",
                "plot_before_final_fill",
                "A large fraction of values are near zero, but the column also contains broader variation.",
            )
        if has_exact_bound:
            return (
                "localized_bound_values_review",
                "mask_exact_bound_values_if_confirmed",
                "Only a smaller number of exact physical-bound values were detected.",
            )
        if has_long_constant:
            return (
                "long_constant_review",
                "plot_before_final_fill",
                "A long constant run was detected, but the full column is not dominated by near-zero values.",
            )

    if has_long_constant:
        return (
            "long_constant_review",
            "plot_before_final_fill",
            "A long constant run was detected.",
        )

    return (
        "manual_review",
        "plot_before_final_fill",
        "The final QC flags require manual interpretation.",
    )


def main() -> None:
    args = parse_args()
    input_path = FINAL_QC_DIR / "final_qc_suspicious_sensors.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run final_qc_summary.py first.")

    suspicious = pd.read_csv(input_path)
    rows: List[dict] = []
    for _, row in suspicious.iterrows():
        decision, action, reason = decide(row, args)
        out = row.to_dict()
        out["QC Decision"] = decision
        out["Recommended Action"] = action
        out["Decision Reason"] = reason
        rows.append(out)

    decisions = pd.DataFrame(rows).sort_values(
        ["QC Decision", "SWC Near-Zero Fraction", "NaN Hours"],
        ascending=[True, False, False],
    )

    summary = (
        decisions.groupby(["QC Decision", "Recommended Action"], dropna=False)
        .agg(
            Rows=("QC Decision", "size"),
            Total_NaN_Hours=("NaN Hours", "sum"),
            Max_Near_Zero_Fraction=("SWC Near-Zero Fraction", "max"),
        )
        .reset_index()
        .sort_values(["Rows", "Total_NaN_Hours"], ascending=False)
    )

    REPORT_DIR.mkdir(exist_ok=True)
    decisions.to_csv(REPORT_DIR / "sensor_qc_decisions.csv", index=False)
    summary.to_csv(REPORT_DIR / "sensor_qc_action_summary.csv", index=False)

    print("Sensor QC decisions complete.")
    print(f"Suspicious rows classified: {len(decisions)}")
    print("Decision counts:")
    print(decisions["QC Decision"].value_counts().to_string())
    print(f"Outputs written under: {REPORT_DIR}")


if __name__ == "__main__":
    main()

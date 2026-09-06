"""Create sensor-level QC candidates and attach explicit human decisions.

This script does not modify station data. It translates the suspicious sensor
metrics from final_qc_summary.py into a candidate table. A candidate remains
pending unless sensor_qc_review_decisions.csv contains a complete approved or
rejected human decision for the same station and parameter.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "sensor_qc_reports"
FINAL_QC_DIR = REPORT_DIR / "before_sensor"
HUMAN_DECISIONS = BASE_DIR / "sensor_qc_review_decisions.csv"
HUMAN_DECISION_COLUMNS = [
    "Station", "Parameter", "Decision", "Reviewer", "Review Date", "Reason",
]
AUTHORIZATION_COLUMNS = [
    "Candidate Status", "Approval Status", "Human Reviewer",
    "Human Review Date", "Human Decision Reason",
]
REQUIRED_CANDIDATE_COLUMNS = {
    "Station", "Parameter", "QC Decision", "Candidate Status",
    "Approval Status", "Human Decision", "Human Reviewer",
    "Human Review Date", "Human Decision Reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build sensor-level QC decision tables.")
    parser.add_argument("--near-zero-bad", type=float, default=0.9)
    parser.add_argument("--near-zero-review", type=float, default=0.5)
    parser.add_argument("--min-hours", type=int, default=720)
    parser.add_argument("--input-dir", type=Path, default=FINAL_QC_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--human-decisions",
        type=Path,
        default=HUMAN_DECISIONS,
        help="Version-controlled approved/rejected whole-sensor decisions.",
    )
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
                "await_human_sensor_decision",
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


def load_human_decisions(path: Path = HUMAN_DECISIONS) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing human sensor decision file: {path}")
    decisions = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(HUMAN_DECISION_COLUMNS) - set(decisions.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )
    decisions = decisions[HUMAN_DECISION_COLUMNS].copy()
    if decisions.empty:
        return decisions

    for column in HUMAN_DECISION_COLUMNS:
        decisions[column] = decisions[column].astype(str).str.strip()
    invalid_decisions = ~decisions["Decision"].isin({"approved", "rejected"})
    if invalid_decisions.any():
        values = sorted(decisions.loc[invalid_decisions, "Decision"].unique())
        raise ValueError(
            "Sensor decisions must be 'approved' or 'rejected'; "
            f"invalid values: {values}"
        )
    required_metadata = ["Station", "Parameter", "Reviewer", "Review Date", "Reason"]
    missing_metadata = decisions[required_metadata].eq("").any(axis=1)
    if missing_metadata.any():
        rows = list(decisions.index[missing_metadata] + 2)
        raise ValueError(
            "Approved/rejected sensor decisions require station, parameter, "
            f"reviewer, review date, and reason; invalid CSV row(s): {rows}"
        )
    parsed_dates = pd.to_datetime(decisions["Review Date"], errors="coerce")
    if parsed_dates.isna().any():
        rows = list(decisions.index[parsed_dates.isna()] + 2)
        raise ValueError(f"Invalid sensor review date at CSV row(s): {rows}")
    duplicate = decisions.duplicated(["Station", "Parameter"], keep=False)
    if duplicate.any():
        pairs = decisions.loc[duplicate, ["Station", "Parameter"]].drop_duplicates()
        raise ValueError(
            "Multiple human sensor decisions exist for: "
            + ", ".join(f"{row.Station}/{row.Parameter}" for row in pairs.itertuples())
        )
    decisions["Review Date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    return decisions


def validate_candidate_authorizations(
    decisions: pd.DataFrame,
    source_name: str = "sensor candidate table",
) -> pd.DataFrame:
    missing = REQUIRED_CANDIDATE_COLUMNS - set(decisions.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing sensor authorization columns: {sorted(missing)}"
        )
    is_candidate = decisions["QC Decision"].eq("bad_sensor_candidate")
    expected_candidate_status = is_candidate.map(
        {True: "candidate", False: "not_candidate"}
    )
    if not decisions["Candidate Status"].equals(expected_candidate_status):
        raise ValueError(
            "Candidate Status must be 'candidate' only for bad_sensor_candidate rows."
        )

    candidates = decisions[is_candidate]
    missing_identity = candidates[["Station", "Parameter"]].isna().any(axis=1)
    missing_identity |= candidates[["Station", "Parameter"]].astype(str).apply(
        lambda column: column.str.strip().eq("")
    ).any(axis=1)
    if missing_identity.any():
        raise ValueError("Sensor candidates require a station and parameter.")
    invalid_status = ~candidates["Approval Status"].isin(
        {"pending", "approved", "rejected"}
    )
    if invalid_status.any():
        raise ValueError("Candidate approval status must be pending, approved, or rejected.")

    noncandidates = decisions[~is_candidate]
    if not noncandidates["Approval Status"].eq("not_applicable").all():
        raise ValueError("Non-candidate rows must use Approval Status 'not_applicable'.")

    pending = candidates[candidates["Approval Status"].eq("pending")]
    pending_decisions = pending["Human Decision"].fillna("").astype(str).str.strip()
    if pending_decisions.ne("").any():
        raise ValueError("Pending candidates cannot contain a human decision.")

    decided = candidates[candidates["Approval Status"].isin({"approved", "rejected"})]
    human_decision = decided["Human Decision"].fillna("").astype(str).str.strip()
    if not human_decision.equals(decided["Approval Status"].astype(str)):
        raise ValueError("Human Decision must match the approved/rejected approval status.")
    metadata = ["Human Reviewer", "Human Review Date", "Human Decision Reason"]
    missing_metadata = decided[metadata].isna().any(axis=1)
    missing_metadata |= decided[metadata].astype(str).apply(
        lambda column: column.str.strip().eq("")
    ).any(axis=1)
    invalid_dates = pd.to_datetime(
        decided["Human Review Date"], errors="coerce"
    ).isna()
    if missing_metadata.any() or invalid_dates.any():
        raise ValueError(
            "Approved/rejected sensor candidates require reviewer, review date, and reason."
        )
    return decisions


def build_decision_table(
    suspicious: pd.DataFrame,
    human_decisions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    human_decisions = human_decisions.copy()
    rows: List[dict] = []
    for _, row in suspicious.iterrows():
        decision, action, reason = decide(row, args)
        out = row.to_dict()
        out["QC Decision"] = decision
        out["Recommended Action"] = action
        out["Decision Reason"] = reason
        out["Candidate Status"] = (
            "candidate" if decision == "bad_sensor_candidate" else "not_candidate"
        )
        rows.append(out)

    decisions = pd.DataFrame(rows) if rows else suspicious.iloc[0:0].copy()
    for column in ["Decision", "Reviewer", "Review Date", "Reason"]:
        if column not in human_decisions:
            human_decisions[column] = pd.Series(dtype=str)
    human = human_decisions.rename(
        columns={
            "Decision": "Human Decision",
            "Reviewer": "Human Reviewer",
            "Review Date": "Human Review Date",
            "Reason": "Human Decision Reason",
        }
    )
    if decisions.empty:
        if not human_decisions.empty:
            raise ValueError(
                "Human sensor decisions exist, but the current QC report contains "
                "no matching candidates."
            )
        for column in [
            "QC Decision", "Recommended Action", "Decision Reason",
            *AUTHORIZATION_COLUMNS,
        ]:
            decisions[column] = pd.Series(dtype=str)
        decisions["Human Decision"] = pd.Series(dtype=str)
        return validate_candidate_authorizations(decisions)

    decisions["Station"] = decisions["Station"].astype(str)
    candidate_pairs = set(
        map(
            tuple,
            decisions.loc[
                decisions["QC Decision"].eq("bad_sensor_candidate"),
                ["Station", "Parameter"],
            ].astype(str).to_numpy(),
        )
    )
    human_pairs = set(
        map(tuple, human_decisions[["Station", "Parameter"]].astype(str).to_numpy())
    )
    unmatched = sorted(human_pairs - candidate_pairs)
    if unmatched:
        raise ValueError(
            "Human whole-sensor decisions do not match current candidates: "
            + ", ".join(f"{station}/{parameter}" for station, parameter in unmatched)
        )
    decisions = decisions.merge(
        human,
        on=["Station", "Parameter"],
        how="left",
        validate="one_to_one",
    )
    candidate = decisions["QC Decision"].eq("bad_sensor_candidate")
    decisions["Approval Status"] = "not_applicable"
    decisions.loc[candidate, "Approval Status"] = (
        decisions.loc[candidate, "Human Decision"].fillna("pending")
    )
    decisions = decisions.sort_values(
        ["QC Decision", "Approval Status", "SWC Near-Zero Fraction", "NaN Hours"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    return validate_candidate_authorizations(decisions)


def main() -> None:
    args = parse_args()
    input_path = args.input_dir / "final_qc_suspicious_sensors.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run final_qc_summary.py first.")

    suspicious = pd.read_csv(input_path)
    human_decisions = load_human_decisions(args.human_decisions)
    decisions = build_decision_table(suspicious, human_decisions, args)

    if not decisions.empty:
        summary = (
            decisions.groupby(
                ["QC Decision", "Approval Status", "Recommended Action"],
                dropna=False,
            )
            .agg(
                Rows=("QC Decision", "size"),
                Total_NaN_Hours=("NaN Hours", "sum"),
                Max_Near_Zero_Fraction=("SWC Near-Zero Fraction", "max"),
            )
            .reset_index()
            .sort_values(["Rows", "Total_NaN_Hours"], ascending=False)
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "QC Decision", "Approval Status", "Recommended Action", "Rows",
                "Total_NaN_Hours", "Max_Near_Zero_Fraction",
            ]
        )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(args.report_dir / "sensor_qc_decisions.csv", index=False)
    summary.to_csv(args.report_dir / "sensor_qc_action_summary.csv", index=False)
    pending = decisions.loc[
        decisions.get("Approval Status", pd.Series(dtype=str)).eq("pending")
    ]
    pending.to_csv(args.report_dir / "sensor_qc_unresolved_candidates.csv", index=False)

    print("Sensor QC decisions complete.")
    print(f"Suspicious rows classified: {len(decisions)}")
    print("Decision counts:")
    print(decisions["QC Decision"].value_counts().to_string())
    print(f"Unresolved whole-sensor candidates: {len(pending)}")
    print(f"Outputs written under: {args.report_dir}")


if __name__ == "__main__":
    main()

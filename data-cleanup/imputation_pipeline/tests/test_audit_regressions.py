from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parents[1]
VISUALIZATION_DIR = REPO_ROOT / "data_visualization"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(VISUALIZATION_DIR))

import FinalResidualGaps
import Longgaps
import MetGaps
import VeryLongGaps
import apply_manual_qc_masks
import apply_sensor_qc_masks
import datacleaning
import final_qc_summary
import imputation_pipeline
import param_config
import sensor_qc_decisions
import txson33_dynamic_visualization
import validate_longgaps
import validate_verylonggaps
from time_index_utils import require_unique_datetime_index


class StrictStagePrerequisiteTests(unittest.TestCase):
    def test_earlier_outputs_do_not_satisfy_validated_stage_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "StationCB01_filled_mediumgaps.csv").touch()
            (output / "StationCB01_filled_longgaps.csv").touch()

            with self.assertRaisesRegex(FileNotFoundError, "validated medium-gap input"):
                Longgaps.input_path_for("CB01", output)
            with self.assertRaisesRegex(FileNotFoundError, "validated long-gap input"):
                VeryLongGaps.input_path_for("CB01", output)

            with mock.patch.object(validate_longgaps, "OUT_DIR", output):
                with self.assertRaisesRegex(FileNotFoundError, "validated medium-gap input"):
                    validate_longgaps.source_path_for("CB01")
            with mock.patch.object(validate_verylonggaps, "OUT_DIR", output):
                with self.assertRaisesRegex(FileNotFoundError, "validated long-gap input"):
                    validate_verylonggaps.source_path_for("CB01")

    def test_qc_stages_require_their_exact_predecessors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "StationCB01_filled_longgaps_repaired.csv").touch()

            with mock.patch.object(apply_sensor_qc_masks, "OUT_DIR", output):
                with self.assertRaisesRegex(FileNotFoundError, "validated very-long-gap input"):
                    apply_sensor_qc_masks.input_path_for("CB01")

            (output / "StationCB01_filled_verylonggaps_repaired.csv").touch()
            with mock.patch.object(apply_manual_qc_masks, "OUT_DIR", output):
                with self.assertRaisesRegex(FileNotFoundError, "sensor-QC input"):
                    apply_manual_qc_masks.input_path_for("CB01")

    def test_final_residual_uses_the_decision_defined_qc_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            sensor = output / "StationCB01_filled_sensor_qc.csv"
            sensor.touch()

            self.assertEqual(
                FinalResidualGaps.input_path_for("CB01", set(), output),
                sensor,
            )
            with self.assertRaisesRegex(FileNotFoundError, "manual-QC input"):
                FinalResidualGaps.input_path_for("CB01", {"CB01"}, output)

            manual = output / "StationCB01_filled_manual_qc.csv"
            manual.touch()
            self.assertEqual(
                FinalResidualGaps.input_path_for("CB01", {"CB01"}, output),
                manual,
            )

    def test_final_qc_runner_selects_explicit_input_stages(self) -> None:
        args = argparse.Namespace(
            station=["CB01"],
            param=None,
            soil_base_dir=Path("soil"),
            met_base_dir=Path("met"),
        )
        names = ["qc-before-sensor", "qc-after-sensor", "qc-final"]
        commands = {
            step.name: step.command
            for step in imputation_pipeline.build_steps(args, names, ["CB01"])
        }
        expected = {
            "qc-before-sensor": "verylong-repaired",
            "qc-after-sensor": "post-qc",
            "qc-final": "final",
        }
        for name, stage in expected.items():
            position = commands[name].index("--input-stage")
            self.assertEqual(commands[name][position + 1], stage)

    def test_sensor_candidates_use_dedicated_presensor_report(self) -> None:
        args = argparse.Namespace(
            station=None,
            param=None,
            soil_base_dir=Path("soil"),
            met_base_dir=Path("met"),
        )
        commands = {
            step.name: step.command
            for step in imputation_pipeline.build_steps(
                args, ["qc-before-sensor", "sensor-decisions"], ["CB01"]
            )
        }
        before = commands["qc-before-sensor"]
        before_dir = before[before.index("--report-dir") + 1]
        sensor = commands["sensor-decisions"]
        sensor_input = sensor[sensor.index("--input-dir") + 1]
        self.assertEqual(before_dir, str(imputation_pipeline.BEFORE_SENSOR_REPORT_DIR))
        self.assertEqual(sensor_input, before_dir)

    def test_post_qc_path_requires_manual_decision_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "StationCB01_filled_sensor_qc.csv").touch()
            with self.assertRaisesRegex(ValueError, "manual-QC station decision set"):
                final_qc_summary.input_path_for("CB01", "post-qc", directory=output)


class ManualOverrideIntervalTests(unittest.TestCase):
    def test_override_is_limited_to_approved_interval(self) -> None:
        run = pd.date_range("2020-01-01", periods=10, freq="h")
        overrides = pd.DataFrame(
            {
                "Station": ["CB20"],
                "Parameter": ["SWC_50"],
                "Start": [run[3]],
                "End": [run[5]],
                "Refill Method": ["donor_mean"],
            }
        )

        segments = FinalResidualGaps.split_run_by_refill_override(
            run, overrides, "CB20", "SWC_50"
        )

        self.assertEqual(
            [(segment[0], segment[-1], method) for segment, method in segments],
            [
                (run[0], run[2], "auto"),
                (run[3], run[5], "donor_mean"),
                (run[6], run[9], "auto"),
            ],
        )

    def test_overlapping_identical_overrides_remain_one_policy_segment(self) -> None:
        run = pd.date_range("2020-01-01", periods=8, freq="h")
        overrides = pd.DataFrame(
            {
                "Station": ["CB20", "CB20"],
                "Parameter": ["SWC_50", "SWC_50"],
                "Start": [run[2], run[4]],
                "End": [run[4], run[6]],
                "Refill Method": ["donor_mean", "donor_mean"],
            }
        )
        segments = FinalResidualGaps.split_run_by_refill_override(
            run, overrides, "CB20", "SWC_50"
        )
        self.assertEqual([method for _, method in segments], ["auto", "donor_mean", "auto"])
        self.assertTrue(segments[1][0].equals(run[2:7]))


class MetTimestampContiguityTests(unittest.TestCase):
    def test_sparse_nan_timestamps_are_separate_runs(self) -> None:
        index = pd.DatetimeIndex(
            ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-10 00:00"]
        )
        series = pd.Series(np.nan, index=index)
        runs = MetGaps.contiguous_nan_runs(series)
        self.assertEqual([list(run) for run in runs], [list(index[:2]), [index[2]]])

    def test_nonmissing_hour_splits_nan_runs(self) -> None:
        index = pd.date_range("2020-01-01", periods=4, freq="h")
        series = pd.Series([np.nan, np.nan, 1.0, np.nan], index=index)
        runs = MetGaps.contiguous_nan_runs(series)
        self.assertEqual([len(run) for run in runs], [2, 1])


class ParameterValidationTests(unittest.TestCase):
    def test_unknown_parameter_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown parameter 'SWC5'"):
            param_config.get_family("SWC5")
        with self.assertRaisesRegex(ValueError, "Unknown parameter 'Tairr'"):
            param_config.short_interp_for("Tairr")

    def test_known_parameter_mapping_is_unchanged(self) -> None:
        self.assertEqual(param_config.get_family("SWC_5"), param_config.SOIL_MOISTURE)
        self.assertEqual(param_config.short_interp_for("Wind direction"), "wind_angle")


class FinalVisualizationInputTests(unittest.TestCase):
    @staticmethod
    def _write_series(path: Path, values: dict[str, list[float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(values, index=pd.date_range("2020-01-01", periods=1, freq="h"))
        frame.index.name = "Date"
        frame.to_csv(path)

    def test_earlier_soil_stage_is_not_used_as_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_series(
                base / "output" / "StationCB01_filled_sensor_qc.csv",
                {"SWC_5": [0.2]},
            )
            self._write_series(
                base / "met_output" / "StationCB01_met_filled_complete.csv",
                {column: [1.0] for column in txson33_dynamic_visualization.MET_COLS},
            )
            with mock.patch.object(txson33_dynamic_visualization, "pipeline_dir", return_value=base):
                with self.assertRaisesRegex(FileNotFoundError, "Final visualization inputs"):
                    txson33_dynamic_visualization.load_station_data("CB01")

    def test_final_met_replaces_stale_met_columns_in_soil_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_series(
                base / "output" / "StationCB01_filled_final.csv",
                {"SWC_5": [0.2], "Ppt": [999.0]},
            )
            met_values = {column: [1.0] for column in txson33_dynamic_visualization.MET_COLS}
            self._write_series(
                base / "met_output" / "StationCB01_met_filled_complete.csv",
                met_values,
            )
            with mock.patch.object(txson33_dynamic_visualization, "pipeline_dir", return_value=base):
                combined = txson33_dynamic_visualization.load_station_data("CB01")
            self.assertEqual(float(combined.iloc[0]["Ppt"]), 1.0)
            self.assertEqual(float(combined.iloc[0]["SWC_5"]), 0.2)

    def test_station_discovery_rejects_unpaired_final_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._write_series(
                base / "output" / "StationCB01_filled_final.csv",
                {"SWC_5": [0.2]},
            )
            with mock.patch.object(txson33_dynamic_visualization, "pipeline_dir", return_value=base):
                with self.assertRaisesRegex(FileNotFoundError, "paired Soil and MET"):
                    txson33_dynamic_visualization.discover_stations()


class SensorCandidateAuthorizationTests(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            near_zero_bad=0.9,
            near_zero_review=0.5,
            min_hours=720,
        )

    @staticmethod
    def _suspicious() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Station": "CB01",
                    "Parameter": "SWC_5",
                    "Flags": "long_constant_run;swc_near_zero_dominant",
                    "SWC Near-Zero Fraction": 0.99,
                    "Nonmissing Hours": 1000,
                    "NaN Hours": 0,
                    "Review Status": "closed",
                    "Review Date": "2026-08-01",
                }
            ]
        )

    @staticmethod
    def _human(decision: str | None = None) -> pd.DataFrame:
        if decision is None:
            return pd.DataFrame(columns=sensor_qc_decisions.HUMAN_DECISION_COLUMNS)
        return pd.DataFrame(
            [
                {
                    "Station": "CB01",
                    "Parameter": "SWC_5",
                    "Decision": decision,
                    "Reviewer": "Reviewer Name",
                    "Review Date": "2026-09-06",
                    "Reason": "Reviewed against the source record.",
                }
            ]
        )

    @staticmethod
    def _sensor() -> pd.DataFrame:
        return pd.DataFrame(
            {"SWC_5": [0.001, np.nan, 0.003]},
            index=pd.date_range("2020-01-01", periods=3, freq="h"),
        )

    def _decision_row(self, human: pd.DataFrame) -> pd.Series:
        table = sensor_qc_decisions.build_decision_table(
            self._suspicious(), human, self._args()
        )
        return table.iloc[0]

    def test_pending_candidate_does_not_mask(self) -> None:
        row = self._decision_row(self._human())
        self.assertEqual(row["Approval Status"], "pending")
        mask = apply_sensor_qc_masks.mask_for_row(self._sensor(), row, False)
        self.assertFalse(mask.any())

    def test_approved_candidate_masks_exact_nonmissing_sensor_values(self) -> None:
        row = self._decision_row(self._human("approved"))
        mask = apply_sensor_qc_masks.mask_for_row(self._sensor(), row, False)
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_rejected_candidate_does_not_mask(self) -> None:
        row = self._decision_row(self._human("rejected"))
        mask = apply_sensor_qc_masks.mask_for_row(self._sensor(), row, False)
        self.assertFalse(mask.any())

    def test_missing_or_invalid_human_metadata_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.csv"
            pd.DataFrame({"Station": ["CB01"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                sensor_qc_decisions.load_human_decisions(path)

            invalid = self._human("approved")
            invalid.loc[0, "Reviewer"] = ""
            invalid.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "require station, parameter"):
                sensor_qc_decisions.load_human_decisions(path)

    def test_inconsistent_generated_authorization_fails_closed(self) -> None:
        table = sensor_qc_decisions.build_decision_table(
            self._suspicious(), self._human(), self._args()
        )
        table.loc[0, "Candidate Status"] = "not_candidate"
        with self.assertRaisesRegex(ValueError, "Candidate Status"):
            sensor_qc_decisions.validate_candidate_authorizations(table)

        table = sensor_qc_decisions.build_decision_table(
            self._suspicious(), self._human(), self._args()
        )
        table.loc[0, "Human Decision"] = "approved"
        with self.assertRaisesRegex(ValueError, "Pending candidates"):
            sensor_qc_decisions.validate_candidate_authorizations(table)

    def test_candidate_generation_is_deterministic(self) -> None:
        first = sensor_qc_decisions.build_decision_table(
            self._suspicious(), self._human(), self._args()
        )
        second = sensor_qc_decisions.build_decision_table(
            self._suspicious(), self._human(), self._args()
        )
        pd.testing.assert_frame_equal(first, second)

    def test_empty_candidate_input_produces_a_valid_empty_table(self) -> None:
        suspicious = self._suspicious().iloc[0:0]
        table = sensor_qc_decisions.build_decision_table(
            suspicious, self._human(), self._args()
        )
        self.assertTrue(table.empty)
        sensor_qc_decisions.validate_candidate_authorizations(table)

    def test_final_qc_reports_pending_candidate_once(self) -> None:
        suspicious = self._suspicious().assign(**{"Review Status": "unresolved"})
        candidates = sensor_qc_decisions.build_decision_table(
            self._suspicious(), self._human(), self._args()
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            final_qc_summary, "BASE_DIR", Path(tmp)
        ):
            unresolved = final_qc_summary.unresolved_review_count(
                suspicious,
                pd.DataFrame(),
                candidates,
            )
        self.assertEqual(unresolved, 1)


class ProductionQcScopeTests(unittest.TestCase):
    def assert_filtered_production_stage_is_rejected(self, stage: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            production = Path(tmp) / "StationCB01_filled_final.csv"
            global_report = Path(tmp) / "final_qc_overview.csv"
            production.write_bytes(b"production-bytes\n")
            global_report.write_bytes(b"global-report-bytes\n")
            before = production.read_bytes()
            report_before = global_report.read_bytes()
            args = argparse.Namespace(
                stage=stage,
                station=None,
                param=["SWC_5"],
                soil_base_dir=Path(tmp),
                met_base_dir=Path(tmp),
                dry_run=False,
                no_clean_stale=False,
            )
            with mock.patch.object(imputation_pipeline, "parse_args", return_value=args), mock.patch.object(
                imputation_pipeline, "clean_stale_outputs"
            ) as cleanup, mock.patch.object(imputation_pipeline, "run_step") as run_step:
                with self.assertRaisesRegex(ValueError, "full station cohort"):
                    imputation_pipeline.main()
            cleanup.assert_not_called()
            run_step.assert_not_called()
            self.assertEqual(production.read_bytes(), before)
            self.assertEqual(global_report.read_bytes(), report_before)

    def test_parameter_only_qc_is_rejected_before_cleanup(self) -> None:
        self.assert_filtered_production_stage_is_rejected("qc")

    def test_parameter_only_final_is_rejected_before_cleanup(self) -> None:
        self.assert_filtered_production_stage_is_rejected("final")

    def test_filtered_final_qc_requires_nonproduction_report_directory(self) -> None:
        args = argparse.Namespace(
            station=None,
            param=["SWC_5"],
            report_dir=final_qc_summary.REPORT_DIR,
        )
        with self.assertRaisesRegex(ValueError, "diagnostic only"):
            final_qc_summary.validate_diagnostic_scope(args)

        args.report_dir = Path("targeted_qc_reports/swc5")
        final_qc_summary.validate_diagnostic_scope(args)


class DuplicateTimestampPolicyTests(unittest.TestCase):
    def test_exact_duplicates_collapse_and_are_counted(self) -> None:
        timestamp = pd.Timestamp("2020-01-01")
        frame = pd.DataFrame(
            {"SWC_5": [0.2, 0.2], "Flag": [0.0, 0.0]},
            index=pd.DatetimeIndex([timestamp, timestamp], name="Date"),
        )
        resolved, summary, conflicts = datacleaning.resolve_duplicate_timestamps(
            frame, "CB01", "soil"
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(summary["Exact Duplicate Groups"], 1)
        self.assertEqual(summary["Rows Removed"], 1)
        self.assertEqual(conflicts, [])

    def test_complementary_duplicates_merge_column_wise(self) -> None:
        timestamp = pd.Timestamp("2020-01-01")
        frame = pd.DataFrame(
            {"SWC_5": [0.2, np.nan], "T_5": [np.nan, 15.0]},
            index=pd.DatetimeIndex([timestamp, timestamp], name="Date"),
        )
        resolved, summary, conflicts = datacleaning.resolve_duplicate_timestamps(
            frame, "CB01", "soil"
        )
        self.assertEqual(float(resolved.loc[timestamp, "SWC_5"]), 0.2)
        self.assertEqual(float(resolved.loc[timestamp, "T_5"]), 15.0)
        self.assertEqual(summary["Complementary Groups"], 1)
        self.assertEqual(conflicts, [])

    def test_measurement_conflict_is_reported_and_set_missing(self) -> None:
        timestamp = pd.Timestamp("2020-01-01")
        frame = pd.DataFrame(
            {"SWC_5": [0.2, 0.3], "T_5": [15.0, 15.0]},
            index=pd.DatetimeIndex([timestamp, timestamp], name="Date"),
        )
        resolved, summary, conflicts = datacleaning.resolve_duplicate_timestamps(
            frame, "CB01", "soil"
        )
        self.assertTrue(pd.isna(resolved.loc[timestamp, "SWC_5"]))
        self.assertEqual(float(resolved.loc[timestamp, "T_5"]), 15.0)
        self.assertEqual(summary["Measurement Conflict Groups"], 1)
        self.assertEqual(conflicts[0]["Conflicting Parameters"], "SWC_5")
        self.assertIn('"SWC_5":0.2', conflicts[0]["Source Rows"])
        self.assertIn('"SWC_5":0.3', conflicts[0]["Source Rows"])

    def test_ambiguous_flag_conflict_is_reported_and_set_missing(self) -> None:
        timestamp = pd.Timestamp("2020-01-01")
        frame = pd.DataFrame(
            {"SWC_5": [0.2, 0.2], "Flag": [0.0, 131074.0]},
            index=pd.DatetimeIndex([timestamp, timestamp], name="Date"),
        )
        resolved, summary, conflicts = datacleaning.resolve_duplicate_timestamps(
            frame, "CB01", "soil"
        )
        self.assertEqual(float(resolved.loc[timestamp, "SWC_5"]), 0.2)
        self.assertTrue(pd.isna(resolved.loc[timestamp, "Flag"]))
        self.assertEqual(summary["Flag-Only Conflict Groups"], 1)
        self.assertEqual(conflicts[0]["Conflict Type"], "flag_only_conflict")

    def test_post_stage_zero_duplicate_index_fails_closed(self) -> None:
        index = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
        frame = pd.DataFrame({"SWC_5": [0.1, 0.2]}, index=index)
        with self.assertRaisesRegex(ValueError, "post-Stage-0 invariant"):
            require_unique_datetime_index(frame, "test input")

    def test_duplicate_report_preserves_summary_and_conflict_values(self) -> None:
        timestamp = pd.Timestamp("2020-01-01")
        frame = pd.DataFrame(
            {"SWC_5": [0.2, 0.3], "Flag": [0.0, 0.0]},
            index=pd.DatetimeIndex([timestamp, timestamp], name="Date"),
        )
        _, summary, conflicts = datacleaning.resolve_duplicate_timestamps(
            frame, "CB01", "soil"
        )
        audit = datacleaning.DuplicateAudit([summary], conflicts)
        with tempfile.TemporaryDirectory() as tmp:
            summary_path, conflict_path = datacleaning.write_duplicate_reports(
                audit, "CB01", Path(tmp)
            )
            written_summary = pd.read_csv(summary_path)
            written_conflicts = pd.read_csv(conflict_path)

        self.assertEqual(written_summary.loc[0, "Station"], "CB01")
        self.assertEqual(written_summary.loc[0, "Source"], "soil")
        self.assertEqual(written_summary.loc[0, "Measurement Conflict Groups"], 1)
        self.assertIn('"SWC_5":0.2', written_conflicts.loc[0, "Source Rows"])
        self.assertIn('"SWC_5":0.3', written_conflicts.loc[0, "Source Rows"])


if __name__ == "__main__":
    unittest.main()

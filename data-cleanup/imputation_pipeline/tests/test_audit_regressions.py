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
import final_qc_summary
import imputation_pipeline
import param_config
import txson33_dynamic_visualization
import validate_longgaps
import validate_verylonggaps


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


if __name__ == "__main__":
    unittest.main()

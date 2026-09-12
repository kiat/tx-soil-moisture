"""Regression tests for timestamp-aligned VeryLong donor holdout metrics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

import VeryLongGaps  # noqa: E402


def series(index: pd.DatetimeIndex, offset: float = 0.0) -> pd.Series:
    values = np.linspace(0.0, 10.0, len(index))
    return pd.Series(values + offset, index=index)


def old_matched_index_metrics(target: pd.Series, donor: pd.Series) -> tuple[float, float]:
    """Previous implementation, valid only when both indexes are identical."""
    mask = target.notna() & donor.notna()
    if int(mask.sum()) < 100:
        return float("nan"), float("nan")
    overlap_idx = target.index[mask].sort_values()
    split = max(1, int(len(overlap_idx) * 0.9))
    train_idx = overlap_idx[:split]
    test_idx = overlap_idx[split:]
    model = LinearRegression().fit(
        donor.loc[train_idx].values.reshape(-1, 1), target.loc[train_idx]
    )
    predicted = model.predict(donor.loc[test_idx].values.reshape(-1, 1))
    return (
        float(mean_absolute_error(target.loc[test_idx], predicted)),
        float(mean_squared_error(target.loc[test_idx], predicted, squared=False)),
    )


class VeryLongHoldoutAlignmentTests(unittest.TestCase):
    def test_identical_timestamps(self):
        index = pd.date_range("2020-01-01", periods=200, freq="h")
        donor = series(index)
        target = donor * 2.0 + 3.0

        mae, rmse = VeryLongGaps.holdout_metrics(target, donor)

        self.assertAlmostEqual(mae, 0.0, places=10)
        self.assertAlmostEqual(rmse, 0.0, places=10)

    def test_target_longer_than_donor(self):
        target_index = pd.date_range("2020-01-01", periods=240, freq="h")
        donor_index = target_index[30:190]
        donor = series(donor_index)
        target = pd.Series(np.nan, index=target_index)
        target.loc[donor_index] = donor * 1.5 + 1.0

        mae, rmse = VeryLongGaps.holdout_metrics(target, donor)

        self.assertTrue(np.isfinite(mae))
        self.assertTrue(np.isfinite(rmse))

    def test_donor_longer_than_target(self):
        donor_index = pd.date_range("2020-01-01", periods=240, freq="h")
        target_index = donor_index[40:200]
        donor = series(donor_index)
        target = donor.reindex(target_index) * 0.75 - 2.0

        mae, rmse = VeryLongGaps.holdout_metrics(target, donor)

        self.assertTrue(np.isfinite(mae))
        self.assertTrue(np.isfinite(rmse))

    def test_partially_overlapping_ranges_use_timestamp_intersection(self):
        target_index = pd.date_range("2020-01-01", periods=220, freq="h")
        donor_index = pd.date_range("2020-01-05", periods=220, freq="h")
        common = target_index.intersection(donor_index)
        donor = series(donor_index)
        target = pd.Series(np.nan, index=target_index)
        target.loc[common] = donor.loc[common] * 2.0 + 4.0

        mae, rmse = VeryLongGaps.holdout_metrics(target, donor)

        self.assertGreaterEqual(len(common), 100)
        self.assertAlmostEqual(mae, 0.0, places=10)
        self.assertAlmostEqual(rmse, 0.0, places=10)

    def test_donor_missing_values_are_excluded_before_chronological_split(self):
        index = pd.date_range("2020-01-01", periods=180, freq="h")
        donor = series(index)
        target = donor * 1.2 + 0.5
        donor.iloc[160:175] = np.nan

        mae, rmse = VeryLongGaps.holdout_metrics(target, donor)

        self.assertTrue(np.isfinite(mae))
        self.assertTrue(np.isfinite(rmse))
        self.assertAlmostEqual(mae, 0.0, places=10)
        self.assertAlmostEqual(rmse, 0.0, places=10)

    def test_no_overlapping_valid_holdout_timestamps(self):
        target_index = pd.date_range("2020-01-01", periods=150, freq="h")
        donor_index = pd.date_range("2021-01-01", periods=150, freq="h")

        mae, rmse = VeryLongGaps.holdout_metrics(
            series(target_index), series(donor_index)
        )

        self.assertTrue(np.isnan(mae))
        self.assertTrue(np.isnan(rmse))

    def test_matches_previous_results_when_indexes_match(self):
        index = pd.date_range("2020-01-01", periods=200, freq="h")
        donor = series(index)
        target = donor * 1.7 + np.sin(np.arange(len(index)) / 12.0)
        donor.iloc[[4, 19, 150]] = np.nan
        target.iloc[[8, 45, 151]] = np.nan

        expected = old_matched_index_metrics(target, donor)
        actual = VeryLongGaps.holdout_metrics(target, donor)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for Soil model-driver missing-value handling."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Longgaps
import Mediumgaps
from soil_source_coverage import SoilCoverage


class SoilDriverMissingnessTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-01-01", periods=240, freq="h", name="Date")

    def test_medium_preserves_missing_drivers_and_observed_zeros(self):
        frame = pd.DataFrame(
            {
                "Ppt": [0.0, np.nan, 1.0],
                "Tair": [0.0, np.nan, 12.0],
                "Srad": [0.0, 20.0, np.nan],
            },
            index=self.index[:3],
        )
        exog = Mediumgaps.get_exog(frame, prefer=("Ppt", "Tair", "Srad"))

        self.assertEqual(exog.loc[self.index[0], "Ppt"], 0.0)
        self.assertEqual(exog.loc[self.index[0], "Tair"], 0.0)
        self.assertEqual(exog.loc[self.index[0], "Srad"], 0.0)
        self.assertTrue(exog.loc[self.index[1], ["Ppt", "Tair"]].isna().all())
        self.assertTrue(pd.isna(exog.loc[self.index[2], "Srad"]))

        first = Mediumgaps.prepare_exog(exog, self.index[:2], self.index[2:3])
        second = Mediumgaps.prepare_exog(exog, self.index[:2], self.index[2:3])
        self.assertIsNone(first[0])
        self.assertIsNone(first[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[2]["Driver Mode"], "univariate_missing_drivers")
        self.assertEqual(first[2]["Drivers Excluded"], "Ppt+Tair+Srad")

    def test_medium_uses_only_complete_driver_and_reports_fallback(self):
        exog = pd.DataFrame(
            {
                "Tair": np.arange(30, dtype=float),
                "Srad": np.arange(30, dtype=float),
            },
            index=self.index[:30],
        )
        exog.loc[self.index[5], "Tair"] = np.nan
        train_index = self.index[:24]
        pred_index = self.index[24:30]

        X_train, X_pred, report = Mediumgaps.prepare_exog(
            exog, train_index, pred_index
        )
        self.assertEqual(list(X_train.columns), ["Srad"])
        self.assertEqual(list(X_pred.columns), ["Srad"])
        self.assertFalse(X_train.isna().any().any())
        self.assertEqual(report["Driver Mode"], "partial_exogenous")
        self.assertEqual(report["Drivers Used"], "Srad")
        self.assertEqual(report["Drivers Excluded"], "Tair")
        self.assertEqual(report["Driver Missing Training"], "Tair=1;Srad=0")

    def test_medium_fill_detail_records_deterministic_driver_mode(self):
        series = pd.Series(0.2, index=self.index)
        series.loc[self.index[100:124]] = np.nan
        coverage = SoilCoverage.from_baseline(
            "TEST", pd.DataFrame({"SWC_5": series.fillna(0.2)}, index=self.index)
        )
        gap = pd.DataFrame(
            {
                "Start Timestamp": [self.index[100]],
                "End Timestamp": [self.index[123]],
            }
        )
        exog = pd.DataFrame({"Ppt": 0.0}, index=self.index)
        exog.loc[self.index[110], "Ppt"] = np.nan
        forecast = pd.Series(0.21, index=self.index[100:124])
        log = []

        with mock.patch.object(
            Mediumgaps, "sarima_forecast", return_value=(forecast, object())
        ):
            Mediumgaps.fill_medium_gaps(
                series, gap, exog, log, "TEST", "SWC_5", coverage=coverage
            )

        self.assertEqual(len(log), 24)
        self.assertEqual({row["Driver Mode"] for row in log}, {"univariate_missing_drivers"})
        self.assertEqual({row["Drivers Excluded"] for row in log}, {"Ppt"})
        self.assertEqual({row["Driver Missing Prediction"] for row in log}, {"Ppt=1"})

    def test_long_driver_columns_preserve_missing_and_proxy_gaps(self):
        frame = pd.DataFrame(
            {
                "Ppt": [0.0, np.nan, 1.0],
                "Tair": [0.0, np.nan, 12.0],
                "Srad": [0.0, np.nan, 30.0],
                "T_5": [10.0, 11.0, 12.0],
            },
            index=self.index[:3],
        )
        sources = Longgaps.ensure_driver_columns(frame)
        self.assertEqual(sources, {"Ppt": "Ppt", "Tair": "Tair", "Srad": "Srad"})
        self.assertEqual(frame.loc[self.index[0], ["Ppt_model", "Tair_model", "Srad_model"]].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(frame.loc[self.index[1], ["Ppt_model", "Tair_model", "Srad_model"]].isna().all())

        proxy = pd.DataFrame(
            {
                "Ppt": [0.0, 0.0, 0.0],
                "Tair": [np.nan, np.nan, np.nan],
                "T_5": [10.0, np.nan, 12.0],
            },
            index=self.index[:3],
        )
        sources = Longgaps.ensure_driver_columns(proxy)
        self.assertEqual(sources["Tair"], "soil_temperature_proxy")
        self.assertEqual(proxy.loc[self.index[0], "Tair_model"], 10.0)
        self.assertTrue(pd.isna(proxy.loc[self.index[1], "Tair_model"]))
        self.assertTrue(proxy["Srad_model"].isna().all())

    def test_long_precipitation_features_distinguish_missing_from_zero(self):
        frame = pd.DataFrame(
            {
                "SWC_5": 0.2,
                "Ppt_model": 0.0,
                "Tair_model": 0.0,
                "Srad_model": 0.0,
            },
            index=self.index,
        )
        timestamp = self.index[200]
        complete = Longgaps.make_features(frame, timestamp, "SWC_5")
        self.assertEqual(complete["ppt_sum7d"], 0.0)
        self.assertEqual(complete["ppt_sum24h"], 0.0)
        self.assertEqual(complete["ppt_flag"], 0.0)
        self.assertEqual(complete["temp_mean"], 0.0)
        self.assertEqual(complete["srad_mean"], 0.0)

        frame.loc[self.index[150], "Ppt_model"] = np.nan
        frame.loc[self.index[151], "Tair_model"] = np.nan
        frame.loc[self.index[152], "Srad_model"] = np.nan
        missing = Longgaps.make_features(frame, timestamp, "SWC_5")
        self.assertTrue(pd.isna(missing["ppt_sum7d"]))
        self.assertTrue(pd.isna(missing["temp_mean"]))
        self.assertTrue(pd.isna(missing["srad_mean"]))

    def test_long_native_missing_fallback_is_deterministic_and_reported(self):
        target = 0.2 + 0.02 * np.sin(np.arange(len(self.index)) / 24)
        frame = pd.DataFrame(
            {
                "SWC_5": target,
                "Ppt": 0.0,
                "Tair": 20.0,
                "Srad": np.nan,
            },
            index=self.index,
        )
        frame.loc[self.index[180], "SWC_5"] = np.nan
        baseline = frame.copy()
        baseline.loc[self.index[180], "SWC_5"] = 0.2
        coverage = SoilCoverage.from_baseline("TEST", baseline)
        Longgaps.ensure_driver_columns(frame)

        model_a = Longgaps.train_xgb(frame.copy(), "SWC_5")
        model_b = Longgaps.train_xgb(frame.copy(), "SWC_5")
        features = Longgaps.make_features(frame, self.index[180], "SWC_5").to_frame().T
        np.testing.assert_allclose(model_a.predict(features), model_b.predict(features))

        gap = pd.DataFrame(
            {"Start Timestamp": [self.index[180]], "End Timestamp": [self.index[180]]}
        )
        with contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(Longgaps, "train_xgb", return_value=object()), \
             mock.patch.object(
                 Longgaps,
                 "rolling_fill",
                 return_value=pd.Series([0.2], index=self.index[180:181]),
             ):
            _, detail = Longgaps.fill_long_gaps_xgb_drift(
                frame, gap, "SWC_5", "TEST", coverage=coverage
            )

        self.assertEqual(detail.loc[0, "Driver Handling"], "xgboost_native_missing")
        sources = json.loads(detail.loc[0, "Driver Sources"])
        missing = json.loads(detail.loc[0, "Prediction Feature Rows With Missing Drivers"])
        self.assertEqual(sources["Srad"], "unavailable")
        self.assertGreater(missing["Srad"], 0)


if __name__ == "__main__":
    unittest.main()

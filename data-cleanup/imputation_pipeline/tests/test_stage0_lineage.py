"""Stage 0 source preservation and hourly lineage regressions."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datacleaning as d


class StageZeroLineageTests(unittest.TestCase):
    def test_union_coverage_and_valid_precipitation_precedence(self):
        hours = pd.date_range("2020-01-01", periods=7, freq="h", name="Date")
        soil = pd.DataFrame({"Ppt": [1., 2., 3., 4.], "SWC_5": [.2]*4}, index=hours[1:5])
        met = pd.DataFrame({"Ppt": [7., 9., np.nan, -1., 8.], "Tair": [20.]*5},
                           index=hours[[0, 2, 3, 4, 6]])
        result = d.merge_hourly_sources(soil, met)
        pd.testing.assert_index_equal(result.index, hours)
        np.testing.assert_allclose(result.Ppt, [7., 1., 9., 3., 4., np.nan, 8.], equal_nan=True)
        self.assertTrue(pd.isna(result.loc[hours[0], "SWC_5"]))
        self.assertTrue(pd.isna(result.loc[hours[1], "Tair"]))
        self.assertTrue(result.index.is_unique and result.index.is_monotonic_increasing)

    def test_no_met_and_all_missing_rain_stay_missing(self):
        hours = pd.date_range("2020-01-01", periods=3, freq="h", name="Date")
        soil = pd.DataFrame({"Ppt": [np.nan, np.inf]}, index=hours[[0, 2]])
        result = d.merge_hourly_sources(soil, pd.DataFrame())
        self.assertEqual(len(result), 3)
        self.assertTrue(result.Ppt.isna().all())

    def test_duplicate_resolution_precedes_hourly_rain_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            # Exact duplicate rain must count once; conflicting SWC alone becomes missing.
            pd.DataFrame({
                "Date": ["2020-01-01 00:00"]*3 + ["2020-01-01 00:30", "2020-01-01 01:00"],
                "Ppt": [1., 1., 1., 2., np.nan],
                "SWC_5": [.1, .1, .2, np.nan, .3],
                "T_5": [np.nan, np.nan, 15., 15., 16.],
            }).to_csv(path / "CB01.dat", index=False)
            audit = d.DuplicateAudit()
            result = d.merge_raw_data("CB01", path, path, audit)
            self.assertEqual(result.Ppt.iloc[0], 3.)
            self.assertTrue(pd.isna(result.SWC_5.iloc[0]))
            self.assertEqual(result.T_5.iloc[0], 15.)
            self.assertTrue(pd.isna(result.Ppt.iloc[1]))
            self.assertEqual(audit.summaries[0]["Rows Removed"], 2)
            self.assertEqual(audit.conflicts[0]["Conflicting Parameters"], "SWC_5")

    def test_conflicting_met_rain_uses_soil_fallback(self):
        ts = pd.DatetimeIndex(["2020-01-01"]*2, name="Date")
        met, summary, conflicts = d.resolve_duplicate_timestamps(
            pd.DataFrame({"Ppt": [1., 2.]}, index=ts), "CB04", "met")
        soil = pd.DataFrame({"Ppt": [3.]}, index=pd.DatetimeIndex([ts[0]], name="Date"))
        self.assertEqual(summary["Measurement Conflict Groups"], 1)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(d.merge_hourly_sources(soil, met).Ppt.iloc[0], 3.)

    def test_invalid_subhour_rain_cannot_cancel_valid_total(self):
        frame = pd.DataFrame({"Ppt": [-10., 2., np.nan, np.nan]},
                             index=pd.date_range("2020-01-01", periods=4, freq="30min"))
        result = d.aggregate_observations_to_hourly(frame)
        self.assertEqual(result.Ppt.iloc[0], 2.)
        self.assertTrue(pd.isna(result.Ppt.iloc[1]))

    def test_unresolved_duplicates_or_subhour_merge_fail(self):
        frame = pd.DataFrame({"Ppt": [1., 2.]}, index=pd.to_datetime(["2020-01-01"]*2))
        for function in [d.aggregate_observations_to_hourly, d.build_hourly_cleaned_data]:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                function(frame)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            d.merge_hourly_sources(frame, pd.DataFrame())
        frame.index = pd.date_range("2020-01-01", periods=2, freq="30min")
        with self.assertRaisesRegex(ValueError, "sub-hourly"):
            d.merge_hourly_sources(frame, pd.DataFrame())

    def _check_real_station(self, station):
        # Extract raw TOA5 source rows on a wet MET-only hour, including duplicates.
        # This avoids repeatedly resolving the full multi-year history in unit tests.
        source = d.DEFAULT_TXSON_DATA_DIR
        soil_file = source / f"{station}.dat"
        met_file = source / f"{station}_met.dat"
        soil = pd.read_csv(soil_file, skiprows=d.find_header_row(soil_file, "Date,"))
        soil["Date"] = pd.to_datetime(soil.Date)
        text = d.repair_concatenated_toa5_records(met_file.read_text())
        met = pd.read_csv(io.StringIO(text), skiprows=d.find_header_row(met_file, '"TIMESTAMP"'),
                          low_memory=False).iloc[2:].rename(columns=d.TOA5_MET_RENAME)
        met["Date"] = pd.to_datetime(met.Date)
        met["Ppt"] = pd.to_numeric(met.Ppt, errors="coerce")
        for outside in [False, True]:
            available = met[met.Ppt.gt(0) & ~met.Date.isin(soil.Date)]
            available = available[(available.Date > soil.Date.max()) if outside else
                                  available.Date.between(soil.Date.min(), soil.Date.max())]
            self.assertFalse(available.empty, (station, outside))
            ts = available.Date.iloc[0]
            rows = met[met.Date.eq(ts)][["Date", "Ppt"]]
            self.assertEqual(rows.Ppt.nunique(), 1)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                soil.iloc[[0, -1]].to_csv(path / f"{station}.dat", index=False)
                rows.to_csv(path / f"{station}_met.dat", index=False)
                result = d.merge_raw_data(station, path, path, d.DuplicateAudit())
                self.assertEqual(result.loc[ts, "Ppt"], float(rows.Ppt.iloc[0]))
                self.assertTrue(result.index.is_unique)
                self.assertTrue(result.index.is_monotonic_increasing)
                self.assertEqual(result.index.max(), max(ts, soil.Date.max()))

    def test_cb04_raw_precipitation_preservation(self):
        self._check_real_station("CB04")

    def test_fd03_raw_precipitation_preservation(self):
        self._check_real_station("FD03")


if __name__ == "__main__":
    unittest.main()

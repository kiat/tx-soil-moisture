"""Coverage regression tests use temporary stage files and cheap predictor stubs."""
import contextlib
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datacleaning
import Shortgaps
import Mediumgaps
import Longgaps
import VeryLongGaps
import FinalResidualGaps
import final_qc_summary
import validate_mediumgaps
import validate_longgaps
import validate_verylonggaps
from soil_source_coverage import SoilCoverage, load_soil_coverage


class SoilSourceCoverageTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-01-01", periods=2600, freq="h", name="Date")
        self.frame = pd.DataFrame({"SWC_5": .2, "SWC_10": .3, "T_20": np.nan, "Ppt": 0.}, index=self.index)
        self.frame.loc[self.index[:200].union(self.index[2400:]), "SWC_5"] = np.nan
        self.frame.loc[self.index[:400].union(self.index[2200:]), "SWC_10"] = np.nan
        for start, end in [(220,222),(300,324),(500,668),(900,1620)]:
            self.frame.loc[self.index[start:end], "SWC_5"] = np.nan
        self.coverage = SoilCoverage.from_baseline("TEST", self.frame)
        self.gaps = datacleaning.create_missing_summary_df(datacleaning.find_missing_data(self.frame))

    def test_parameter_bounds_late_start_early_end_absent_and_never_valid(self):
        self.assertEqual(self.coverage.bounds["SWC_5"], (self.index[200], self.index[2399]))
        self.assertEqual(self.coverage.bounds["SWC_10"], (self.index[400], self.index[2199]))
        self.assertFalse(self.coverage.contains(self.index[[399,2200]], "SWC_10").any())
        self.assertEqual(self.coverage.statuses["SWC_50"], "column_absent")
        self.assertEqual(self.coverage.statuses["T_20"], "no_valid_source_observations")
        self.assertFalse(self.coverage.contains(self.index, "T_20").any())
        self.assertFalse(self.coverage.contains(self.index, "SWC_50").any())

    def test_all_four_loaders_restrict_before_classifying(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.gaps.to_csv(folder / "StationTEST_missing_data.csv", index=False)
            tables = [module.load_missing_data("TEST", directory=folder, coverage=self.coverage)
                      for module in [Shortgaps, Longgaps, VeryLongGaps]]
            with mock.patch.object(Mediumgaps, "MISS_DIR", folder):
                tables.append(Mediumgaps.load_missing_data("TEST", coverage=self.coverage))
        for table in tables:
            soil = table[table.Parameter.eq("SWC_5")]
            self.assertEqual(soil["Number Missing"].tolist(), [2,24,168,720])
            self.assertFalse(table.Parameter.isin(["SWC_10","SWC_50","T_20"]).any())
            self.assertEqual(len(Shortgaps.filter_short_gaps(table.copy(), "SWC_5")), 1)
            self.assertEqual(len(Mediumgaps.filter_medium_gaps(table.copy(), "SWC_5")), 1)
            self.assertEqual(len(Longgaps.filter_long_gaps(table.copy(), "SWC_5")), 1)
            self.assertEqual(len(VeryLongGaps.filter_very_long_gaps(table.copy(), "SWC_5")), 1)

    def test_internal_short_medium_long_fill_but_boundaries_stay_nan(self):
        table = self.coverage.restrict_gaps(self.gaps)
        log = []
        short = Shortgaps.fill_short_gaps(self.frame.SWC_5, Shortgaps.filter_short_gaps(table.copy(), "SWC_5"),
            log, station_id="TEST", param="SWC_5", coverage=self.coverage)
        self.assertTrue(short.iloc[220:222].notna().all())
        with mock.patch.object(Mediumgaps, "sarima_forecast", side_effect=lambda y,s,e,*a:
                               (pd.Series(.2,index=pd.date_range(s,e,freq="h")),None)) as predictor:
            medium = Mediumgaps.fill_medium_gaps(self.frame.SWC_5, Mediumgaps.filter_medium_gaps(table.copy(), "SWC_5"),
                None, [], "TEST", "SWC_5", coverage=self.coverage)
            self.assertEqual(predictor.call_count, 1)
        self.assertTrue(medium.iloc[300:324].notna().all())
        with mock.patch.object(Longgaps, "train_xgb", return_value=object()), \
             mock.patch.object(Longgaps, "rolling_fill", side_effect=lambda model,df,idx,param: pd.Series(.2,index=idx)):
            long, _ = Longgaps.fill_long_gaps_xgb_drift(self.frame, Longgaps.filter_long_gaps(table.copy(), "SWC_5"),
                "SWC_5", "TEST", coverage=self.coverage)
        self.assertTrue(long.iloc[500:668].notna().all())
        for result in [short, medium, long]:
            self.assertTrue(result.iloc[:200].isna().all())
            self.assertTrue(result.iloc[2400:].isna().all())

    def test_direct_fill_calls_reject_outside_intervals_before_predicting(self):
        gap = self.gaps[self.gaps.Parameter.eq("SWC_5")].iloc[[0]]
        with self.assertRaisesRegex(ValueError, "outside Soil source coverage"):
            Shortgaps.fill_short_gaps(self.frame.SWC_5,gap,[],station_id="TEST",param="SWC_5",coverage=self.coverage)
        with mock.patch.object(Mediumgaps, "sarima_forecast") as predictor:
            with self.assertRaisesRegex(ValueError, "outside Soil source coverage"):
                Mediumgaps.fill_medium_gaps(self.frame.SWC_5,gap,None,[],"TEST","SWC_5",coverage=self.coverage)
            predictor.assert_not_called()
        with mock.patch.object(Longgaps, "train_xgb") as predictor:
            with self.assertRaisesRegex(ValueError, "outside Soil source coverage"):
                Longgaps.fill_long_gaps_xgb_drift(self.frame,gap,"SWC_5","TEST",coverage=self.coverage)
            predictor.assert_not_called()

    def test_verylong_and_final_override_cannot_fill_outside_coverage(self):
        donor = pd.DataFrame({"SWC_5": .25 + .01*np.sin(np.arange(len(self.index))/24)}, index=self.index)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            with mock.patch.object(VeryLongGaps,"OUT_DIR",directory), \
                 mock.patch.object(VeryLongGaps,"load_soil_coverage",return_value=self.coverage), \
                 mock.patch.object(VeryLongGaps,"load_missing_data",return_value=self.coverage.restrict_gaps(self.gaps)):
                VeryLongGaps.fill_station("TEST",["SWC_5","SWC_50","T_20"],{"TEST":self.frame,"DONOR":donor},24,.3)
            vl = pd.read_csv(directory/"StationTEST_filled_verylonggaps.csv",index_col=0,parse_dates=True)
            self.assertTrue(vl.SWC_5.iloc[900:1620].notna().all())
            # Simulate human QC masking every legitimate target observation.
            target = self.frame.copy()
            target["SWC_5"] = np.nan
            target["SWC_50"] = np.nan  # Even an accidentally introduced empty absent column has no support.
            override = pd.DataFrame({"Station":["TEST"],"Parameter":["SWC_5"],"Start":[self.index[0]],
                                     "End":[self.index[-1]],"Refill Method":["donor_mean"]})
            with mock.patch.object(FinalResidualGaps,"OUT_DIR",directory), \
                 mock.patch.object(FinalResidualGaps,"load_soil_coverage",return_value=self.coverage):
                FinalResidualGaps.fill_station("TEST",["SWC_5","SWC_50","T_20"],
                    {"TEST":target,"DONOR":donor.assign(SWC_50=donor.SWC_5,T_20=20.)},24,.3,override)
            final = pd.read_csv(directory/"StationTEST_filled_final.csv",index_col=0,parse_dates=True)
            self.assertTrue(final.SWC_5.iloc[200:2400].notna().all())
            self.assertTrue(final.SWC_50.isna().all())
            self.assertTrue(final.T_20.isna().all())
            for result in [vl,final]:
                self.coverage.assert_frame(result)
                self.assertTrue(result.SWC_5.iloc[:200].isna().all())
                self.assertTrue(result.SWC_5.iloc[2400:].isna().all())

    def test_gap_report_excludes_outside_and_retains_provenance(self):
        rows = final_qc_summary.nan_runs(self.frame,"TEST","SWC_5",self.coverage)
        self.assertEqual([row['Hours'] for row in rows], [2,24,168,720])
        records = {r['Parameter']:r for r in self.coverage.records(self.frame)}
        self.assertEqual(records['SWC_5']['Internal NaN Hours'],914)
        self.assertEqual(records['SWC_5']['Outside Source Coverage NaN Hours'],400)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(sys,"argv",["final_qc_summary.py","--station","TEST","--param","SWC_5",
                                           "--input-stage","verylong-repaired","--report-dir",tmp]), \
             mock.patch.object(final_qc_summary,"read_station",return_value=self.frame), \
             mock.patch.object(final_qc_summary,"input_path_for",return_value=Path("synthetic.csv")), \
             mock.patch.object(final_qc_summary,"load_soil_coverage",return_value=self.coverage), \
             mock.patch.object(final_qc_summary,"load_review_decisions",return_value=pd.DataFrame()), \
             mock.patch.object(final_qc_summary,"summarize_verylong_review",return_value=pd.DataFrame()), \
             mock.patch.object(final_qc_summary,"unresolved_review_count",return_value=0):
            final_qc_summary.main()
            summary = pd.read_csv(Path(tmp)/"final_qc_station_parameter_summary.csv")
            overview = pd.read_csv(Path(tmp)/"final_qc_overview.csv")
            self.assertEqual(summary.loc[0,'NaN Hours'],914)
            self.assertEqual(summary.loc[0,'Outside Source Coverage NaN Hours'],400)
            self.assertEqual(overview.loc[0,'Remaining Soil NaN Hours'],914)
            self.assertTrue((Path(tmp)/"final_qc_soil_source_coverage.csv").exists())

    def test_validators_expect_only_internal_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.gaps.to_csv(Path(tmp)/"StationTEST_missing_data.csv",index=False)
            for module,size in [(validate_mediumgaps,24),(validate_longgaps,168),(validate_verylonggaps,720)]:
                with mock.patch.object(module,"MISS_DIR",Path(tmp)), \
                     mock.patch.object(module,"load_soil_coverage",return_value=self.coverage):
                    expected = module.load_expected_segments(["TEST"],["SWC_5","SWC_10","T_20"])
                    self.assertEqual(list(expected.values()),[size])

    def test_sparse_hours_are_separate_and_stale_filled_boundary_fails(self):
        frame = self.frame.loc[self.index[[300,301,320]]]
        self.assertEqual([len(r) for r in self.coverage.nan_runs(frame,"SWC_5")],[2,1])
        bad = self.frame.copy();bad.loc[self.index[0],"SWC_5"] = .2
        with self.assertRaisesRegex(ValueError,"outside Soil source coverage"):
            self.coverage.assert_frame(bad)

    def test_stale_donor_outside_coverage_fails_before_any_station_write(self):
        bad = self.frame.copy(); bad.loc[self.index[0], 'SWC_5'] = .2
        data = {'TEST': self.frame, 'OTHER': bad}
        args = argparse.Namespace(station=['TEST'],param=['SWC_5'],min_overlap=24,min_abs_corr=.3)
        for module in [VeryLongGaps,FinalResidualGaps]:
            with self.subTest(stage=module.__name__), contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(module,'parse_args',return_value=args))
                stack.enter_context(mock.patch.object(module,'discover_stations',return_value=list(data)))
                stack.enter_context(mock.patch.object(module,'load_soil_coverage',return_value=self.coverage))
                fill = stack.enter_context(mock.patch.object(module,'fill_station'))
                if module is VeryLongGaps:
                    stack.enter_context(mock.patch.object(module,'load_stage_data',side_effect=data.__getitem__))
                else:
                    stack.enter_context(mock.patch.object(module,'load_manual_masks',return_value=pd.DataFrame(columns=['Station'])))
                    stack.enter_context(mock.patch.object(module,'input_path_for',side_effect=lambda station,*a: station))
                    stack.enter_context(mock.patch.object(module,'read_station',side_effect=data.__getitem__))
                with self.assertRaisesRegex(ValueError,'outside Soil source coverage'):
                    module.main()
                fill.assert_not_called()

    def test_validator_rejects_stale_outside_fill_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            detail = pd.DataFrame({'Station':['TEST'],'Parameter':['SWC_5'],
                'Start':[self.index[0]],'End':[self.index[0]],'Timestamp':[self.index[0]],'Filled':[.2]})
            for module,suffix in [(validate_mediumgaps,'medium'),(validate_longgaps,'long'),(validate_verylonggaps,'verylong')]:
                detail.to_csv(directory/f'StationTEST_{suffix}gap_fill_detail.csv',index=False)
                with mock.patch.object(module,'OUT_DIR',directory), \
                     mock.patch.object(module,'load_soil_coverage',return_value=self.coverage):
                    with self.assertRaisesRegex(ValueError,'outside Soil source coverage'):
                        module.load_filled_segments(['TEST'],['SWC_5'])

    def test_manifest_required_and_tampered_baseline_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp);(base/'cleaned_data').mkdir();(base/'stage0_reports').mkdir()
            path = base/'cleaned_data/StationTEST_cleaned_data.csv';self.frame.to_csv(path)
            with self.assertRaises(FileNotFoundError):load_soil_coverage("TEST",base)
            manifest = {'station':'TEST','outputs':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}
            (base/'stage0_reports/StationTEST_provenance.json').write_text(json.dumps(manifest))
            self.assertEqual(load_soil_coverage('TEST',base).bounds,self.coverage.bounds)
            self.frame.fillna(.2).to_csv(path)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_soil_coverage('TEST',base)

    def test_real_met_union_stations_have_protected_boundaries(self):
        for station in ['WC05','CB04','FD02']:
            coverage = load_soil_coverage(station)
            base = datacleaning.DEFAULT_TXSON_DATA_DIR
            rawfile = base/f'{station}.dat'
            raw = pd.read_csv(rawfile,skiprows=datacleaning.find_header_row(rawfile,'Date,'))
            raw['Date'] = pd.to_datetime(raw.Date)
            frame = pd.read_csv(Path(__file__).resolve().parents[1]/'cleaned_data'/f'Station{station}_cleaned_data.csv',index_col=0,parse_dates=True)
            records = coverage.records(frame)
            for param in ['SWC_5','SWC_50','T_5']:
                values = pd.to_numeric(raw[param],errors='coerce')
                low,high = (0,.6) if param.startswith('SWC') else (-30,60)
                dates = raw.loc[np.isfinite(values)&values.between(low,high),'Date']
                self.assertEqual(coverage.bounds[param],(dates.min(),dates.max()))
            self.assertGreater(sum(r['Outside Source Coverage NaN Hours'] for r in records),0)
            if station=='FD02':self.assertLess(frame.index.min(),coverage.bounds['SWC_5'][0])
            self.assertGreater(frame.index.max(),coverage.bounds['SWC_5'][1])
            for run in coverage.nan_runs(frame,'SWC_5'):
                self.assertTrue(coverage.contains(run,'SWC_5').all())


if __name__ == '__main__':
    unittest.main()

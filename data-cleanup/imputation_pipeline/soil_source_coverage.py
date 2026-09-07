"""Soil imputation support fixed by the verified, pre-imputation Stage 0 baseline."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import json

import numpy as np
import pandas as pd

from param_config import ALL_SOIL_PARAMS
from time_index_utils import require_unique_datetime_index

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SoilCoverage:
    station: str
    bounds: dict
    statuses: dict
    baseline_sha256: str = ""

    @classmethod
    def from_baseline(cls, station, frame, baseline_sha256=""):
        require_unique_datetime_index(frame, "Soil source coverage baseline")
        bounds, statuses = {}, {}
        for param in ALL_SOIL_PARAMS:
            if param not in frame:
                bounds[param], statuses[param] = None, "column_absent"
                continue
            values = frame[param]
            # The baseline already applied duplicate and physical-range QC.
            index = frame.index[np.isfinite(values)]
            bounds[param] = (index.min(), index.max()) if len(index) else None
            statuses[param] = "covered" if len(index) else "no_valid_source_observations"
        return cls(str(station), bounds, statuses, baseline_sha256)

    def contains(self, index, param):
        if param not in self.bounds:
            raise ValueError(f"Unknown Soil parameter: {param}")
        bound = self.bounds[param]
        if bound is None:
            return np.zeros(len(index), dtype=bool)
        return (index >= bound[0]) & (index <= bound[1])

    def require_interval(self, start, end, param):
        if start > end or not self.contains(pd.DatetimeIndex([start, end]), param).all():
            raise ValueError(f"Station{self.station} {param}: fill interval {start} to {end} "
                             "is outside Soil source coverage.")

    def assert_frame(self, frame):
        """Reject stale/invalid stage values rather than silently propagating extrapolations."""
        require_unique_datetime_index(frame, "Soil stage coverage check")
        for param in ALL_SOIL_PARAMS:
            if param in frame and (frame[param].notna() & ~self.contains(frame.index, param)).any():
                raise ValueError(f"Station{self.station} {param}: nonmissing values outside Soil source coverage.")

    def restrict_gaps(self, table):
        """Intersect before gap-length classification; retain the full Stage 0 NaN inventory on disk."""
        result = table.copy()
        for param in ALL_SOIL_PARAMS:
            selected = result.Parameter.eq(param)
            bound = self.bounds[param]
            if bound is None:
                result = result.loc[~selected].copy()
                continue
            result.loc[selected, "Start Timestamp"] = result.loc[selected, "Start Timestamp"].clip(lower=bound[0])
            result.loc[selected, "End Timestamp"] = result.loc[selected, "End Timestamp"].clip(upper=bound[1])
        result = result.loc[result["Start Timestamp"] <= result["End Timestamp"]].copy()
        soil = result.Parameter.isin(ALL_SOIL_PARAMS)
        result.loc[soil, "Number Missing"] = (
            (result.loc[soil, "End Timestamp"] - result.loc[soil, "Start Timestamp"]) / pd.Timedelta(hours=1) + 1
        ).astype(int)
        return result

    def nan_runs(self, frame, param):
        if param not in frame:
            return []
        selected = frame.index[frame[param].isna() & self.contains(frame.index, param)]
        if not len(selected):
            return []
        # Adjacency is measured in time, including when a caller supplies a sparse index.
        breaks = np.flatnonzero((selected[1:] - selected[:-1]) != pd.Timedelta(hours=1)) + 1
        return [pd.DatetimeIndex(run) for run in np.split(selected, breaks)]

    def records(self, frame):
        rows = []
        for param in ALL_SOIL_PARAMS:
            inside = self.contains(frame.index, param)
            present = param in frame
            missing = frame[param].isna().to_numpy() if present else np.zeros(len(frame), dtype=bool)
            bound = self.bounds[param]
            rows.append({
                "Station": self.station, "Parameter": param,
                "Source Coverage Status": self.statuses[param],
                "Source Start": bound[0] if bound else pd.NaT,
                "Source End": bound[1] if bound else pd.NaT,
                "Inside Source Coverage Hours": int(inside.sum()),
                "Internal NaN Hours": int((missing & inside).sum()),
                "Outside Source Coverage NaN Hours": int((missing & ~inside).sum()),
                "Column Present": present, "Coverage Baseline SHA256": self.baseline_sha256,
            })
        return rows


def load_soil_coverage(station, base_dir=None):
    """Require the authoritative Stage 0 manifest; never infer bounds from imputed/QC data."""
    base = Path(base_dir) if base_dir is not None else BASE_DIR
    baseline = base / "cleaned_data" / f"Station{station}_cleaned_data.csv"
    provenance = base / "stage0_reports" / f"Station{station}_provenance.json"
    manifest = json.loads(provenance.read_text())
    if str(manifest["station"]) != str(station):
        raise ValueError(f"Station{station}: Stage 0 provenance station mismatch.")
    entries = [entry for entry in manifest["outputs"] if Path(entry["path"]).name == baseline.name]
    data = baseline.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(entries) != 1 or digest != entries[0]["sha256"]:
        raise ValueError(f"Station{station}: Stage 0 baseline hash mismatch; regenerate/verify Stage 0 first.")
    frame = pd.read_csv(io.BytesIO(data), index_col=0, parse_dates=True)
    # Stage 0 Soil columns come only from Soil raw inputs, never from MET.
    return SoilCoverage.from_baseline(station, frame, digest)

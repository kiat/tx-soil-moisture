# -----------------------------------------------------------
# Mediumgaps.py  -  Fill 24‑ to 168‑hour gaps via SARIMAX
# -----------------------------------------------------------
# usage examples
#    python Mediumgaps.py                       # all stations & all SWC/T columns
#    python Mediumgaps.py --station CB01        # only station CB01
#    python Mediumgaps.py --param SWC_20        # all stations, only SWC_20
#    python Mediumgaps.py --station FD08 --param SWC_50  # specific combo
#
# Output files (per station)
#    output/StationX_filled_mediumgaps.csv       - cleaned series after filling
#    output/StationX_mediumgap_fill_detail.csv   - long‑form log of every value written
# -----------------------------------------------------------

# Import libraries
import warnings
import argparse, re
from pathlib import Path
from datetime import timedelta

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima

from param_config import ALL_SOIL_PARAMS, exog_for

warnings.filterwarnings("ignore")

# Paths 
BASE_DIR  = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "output"                  # short-gap-filled inputs
MISS_DIR  = BASE_DIR / "missing_data"
OUT_DIR   = BASE_DIR / "output"



def load_cleaned_data(station_id):
    filename = CLEAN_DIR / f"Station{station_id}_filled_shortgaps.csv"
    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    return ensure_hourly_regular_index(df)

def load_missing_data(station_id):
    filename = MISS_DIR / f"Station{station_id}_missing_data.csv"
    df = pd.read_csv(filename, parse_dates=["Start Timestamp", "End Timestamp"])
    return df



# Get medium gaps
def filter_medium_gaps(df_missing, parameter="SWC_5", min_gap=24, max_gap=168):
    # Convert "Number Missing" column to numeric
    df_missing["Number Missing"] = pd.to_numeric(df_missing["Number Missing"], errors="coerce")
    mask = (df_missing["Parameter"] == parameter) & \
           (df_missing["Number Missing"] >= min_gap) & (df_missing["Number Missing"] <= max_gap)
    return df_missing.loc[mask].sort_values("Start Timestamp")



# Fit SARIMAX model and predict missing values
def sarima_forecast(y, s_ts, e_ts, exog, ctx_days = 7, max_pq = 3, max_PQ = 2):
    # Determine the training window up to one hour before gap
    train_start = s_ts - timedelta(days=ctx_days)
    train_end = s_ts - timedelta(hours=1)

    # Extract and locally regularize training data. Short gaps are already filled,
    # but longer neighboring gaps can still leave NaNs in this context window.
    y_window = y.loc[train_start:train_end]
    observed = y_window.dropna()
    if len(observed) < 24:
        print(f"[skip] Only {len(observed)} observed training hours {train_start}–{train_end}")
        return None, None
    y_train = y_window.interpolate(method="time", limit_direction="both").ffill().bfill()
    y_train.index = pd.DatetimeIndex(y_train.index, freq="H")

    # Prepare exogenous data
    X_train = X_pred = None
    if exog is not None:
        exog_window = exog.loc[train_start:train_end]
        X_train = exog_window.reindex(y_train.index).fillna(0)
        pred_index = pd.date_range(s_ts, e_ts, freq="H")
        X_pred = exog.reindex(pred_index).fillna(0)

    # Automatic model order selection with daily seasonality
    try:
        auto = auto_arima(
            y_train,
            X=X_train,
            seasonal=True, m=24,
            d=None, D=1,
            start_p=1, start_q=1, max_p=max_pq, max_q=max_pq,
            start_P=0, start_Q=0, max_P=max_PQ, max_Q=max_PQ,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )
        p, d, q = auto.order
        P, D, Q, s = auto.seasonal_order
        print(f"  → SARIMA({p},{d},{q})x({P},{D},{Q},{s})24h")

        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False,
            freq="H"
        )
        res = model.fit(method="powell", maxiter=300, disp=False)

        # Check residuals for autocorrelation when enough residuals are available.
        if len(res.resid) > 24:
            lb_p = acorr_ljungbox(res.resid, lags=[24], return_df=True)["lb_pvalue"].iat[0]
            if lb_p < 0.05 and (max_pq < 5 or max_PQ < 3):
                return sarima_forecast(
                    y, s_ts, e_ts, exog,
                    ctx_days=ctx_days,
                    max_pq=max_pq+1,
                    max_PQ=max_PQ+1
                )

        forecast_index = pd.date_range(s_ts, e_ts, freq="H")
        fc = res.forecast(steps=len(forecast_index), exog=X_pred)
    except Exception as exc:
        print(f"[skip] SARIMAX failed for {s_ts}–{e_ts}: {exc}")
        return None, None

    fc.index = forecast_index
    return fc, res



# Iterate through each medium gap, fit SARIMAX, and write predictions back
def fill_medium_gaps(series, gaps, exog, gap_log, station, param, ctx_days=7):
    filled = series.copy()
    for _, row in gaps.iterrows():
        s_ts = row["Start Timestamp"]
        e_ts = row["End Timestamp"]
        idx = pd.date_range(s_ts, e_ts, freq="H")

        fc, _ = sarima_forecast(filled, s_ts, e_ts, exog, ctx_days)
        if fc is None:
            continue
        fc = correct_boundary_drift(fc, filled, s_ts, e_ts)
        fc = apply_physical_bounds(fc, param)

        filled.loc[idx] = fc
        gap_log.extend({
            "Station": station, "Parameter": param,
            "Start": s_ts, "End": e_ts,
            "Timestamp": t, "Filled": v
        } for t, v in fc.items())
    return filled


def correct_boundary_drift(fc, observed, s_ts, e_ts):
    """Linearly anchor a forecast to real observations around the gap."""
    left_ts = s_ts - timedelta(hours=1)
    right_ts = e_ts + timedelta(hours=1)
    has_left = left_ts in observed.index and pd.notna(observed.loc[left_ts])
    has_right = right_ts in observed.index and pd.notna(observed.loc[right_ts])

    corrected = fc.copy()
    if has_left and has_right:
        left_delta = observed.loc[left_ts] - corrected.iloc[0]
        right_delta = observed.loc[right_ts] - corrected.iloc[-1]
        if len(corrected) == 1:
            corrected.iloc[0] = 0.5 * (observed.loc[left_ts] + observed.loc[right_ts])
        else:
            correction = [
                left_delta + (right_delta - left_delta) * i / (len(corrected) - 1)
                for i in range(len(corrected))
            ]
            corrected = corrected + pd.Series(correction, index=corrected.index)
    elif has_left:
        corrected.iloc[0] = 0.5 * (observed.loc[left_ts] + corrected.iloc[0])
    elif has_right:
        corrected.iloc[-1] = 0.5 * (observed.loc[right_ts] + corrected.iloc[-1])
    return corrected


def apply_physical_bounds(fc, param):
    """Keep model output within basic physical ranges used by cleaning."""
    if isinstance(param, str) and param.startswith("SWC_"):
        return fc.clip(lower=0, upper=0.6)
    if isinstance(param, str) and (param.startswith("T_") or param == "Tair"):
        return fc.clip(lower=-30, upper=60)
    return fc



# Fill medium gaps for each SWC parameter and save outputs
def process_station(station, params):
    df = load_cleaned_data(station)
    miss_tbl = load_missing_data(station)

    # Regularize indices to avoid frequency issues
    df = ensure_hourly_regular_index(df)

    log = []
    filled_count = 0

    for p in params:
        if p not in df.columns:
            print(f"  {p}: column missing in short-gap data, skip.")
            continue
        mgaps = filter_medium_gaps(miss_tbl, p)
        if mgaps.empty:
            print(f"  {p}: no 24–168 h gaps")
            continue
        exog = get_exog(df, prefer=exog_for(p))

        print(f"  {p}: filling {len(mgaps)} gaps")
        before = len(log)
        df[p] = fill_medium_gaps(df[p], mgaps, exog, log, station, p)
        filled_count += len(log) - before

    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / f"Station{station}_filled_mediumgaps.csv")
    if log:
        pd.DataFrame(log).to_csv(
            OUT_DIR / f"Station{station}_mediumgap_fill_detail.csv", index=False)
    status = " (unchanged)" if filled_count == 0 else f" ({filled_count} values filled)"
    print(f"→ saved Station{station}_filled_mediumgaps.csv{status}\n")


# --------------- helpers ---------------------------

def ensure_hourly_regular_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the index is an hourly, continuous DateTimeIndex (no duplicates, sorted).
    Keeps existing values and inserts NaNs for any missing timestamps.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    if len(df.index) == 0:
        return df
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='H')
    return df.reindex(full_idx)


def get_exog(df: pd.DataFrame, prefer=()):
    """
    Build an exogenous variables DataFrame using available preferred columns.
    Returns None when no usable exogenous driver exists.
    """
    exog_series = []
    for col in prefer:
        if col not in df.columns:
            continue
        s = df[col]
        if s.notna().any():
            exog_series.append(s.rename(col))
    if not exog_series:
        return None
    X = pd.concat(exog_series, axis=1).reindex(df.index)
    X = X.fillna(0.0)
    return X


# --------------- CLI helpers ---------------------------

def discover_stations():
    pat = re.compile(r"Station(.+)_filled_shortgaps\.csv")
    return sorted(m.group(1) for fn in CLEAN_DIR.glob("Station*_filled_shortgaps.csv")
                  if (m := pat.match(fn.name)))

def parse_args():
    p = argparse.ArgumentParser("Fill 24–168 h gaps via SARIMAX")
    p.add_argument("--station", type=str, nargs="*", help="station IDs/site codes")
    p.add_argument("--param",   type=str, nargs="*", help="Columns to fill (SWC_* or T_*).")
    return p.parse_args()

# --------------- main entry ----------------------------

def main():
    args = parse_args()
    stations = args.station if args.station else discover_stations()
    # Default: fill both soil moisture and soil temperature medium gaps.
    default_params = ALL_SOIL_PARAMS
    params   = args.param if args.param else default_params

    print("Stations :", stations)
    print("Parameters:", params, "\n")

    for sid in stations:
        print(f"=== Station {sid} ===")
        process_station(sid, params)

if __name__ == "__main__":
    main()

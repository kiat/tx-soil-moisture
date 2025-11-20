# Soil Moisture & Temperature Gap-Filling Pipeline

### Zun Cao
This folder contains the full imputation workflow we use to rebuild continuous soil **moisture** (SWC) and **temperature** (T):

```text
data-cleanup/imputation_pipeline/
    datacleaning.py            # step 0 – build cleaned CSV + gap metadata
    Shortgaps.py               # step 1 – fill <24h gaps (SWC + T)
    Mediumgaps.py              # step 2 – fill 1–7 day gaps via SARIMAX
    Longgaps.py                # step 3 – fill 7–30 day gaps via XGBoost
    VeryLongGaps.py            # step 4 – fill ≥30 day gaps via cross-station regression
    imputation_pipeline.py     # orchestrator
    cleaned_data/              # auto-created outputs from datacleaning
    missing_data/              # auto-created missing summaries
    output/                    # staged gap-filling outputs/logs
```

The scripts now include soil-temperature logic everywhere a fill occurs. Manual overrides for persistent bad-but-present sensors (for example, Station 5 T_20 stuck during 2015–2016) are injected automatically during the cleaning step so every downstream stage respects those spans.

---

## Prerequisites

1. **Python** ≥ 3.10 (matches the repo tooling).  
2. Install dependencies once:

   ```bash
   pip install -r requirements.txt
   ```

3. Raw inputs live under `datasets/TX-Data/soil_station` (`SM_{id}.dat`) and `datasets/TX-Data/met_station` (`MET_{id}.dat`). Datacleaning reads from there by default; override with `--soil-base-dir`/`--met-base-dir` if needed.

---

## Stage-by-Stage Summary

1. **`datacleaning.py`**  
   - Merges soil + MET `.dat` files, enforces physical bounds, and replaces invalid values with `NaN`.  
   - Builds two artifacts per station: `cleaned_data/Station{id}_cleaned_data.csv` (hourly timeline) and `missing_data/Station{id}_missing_data.csv`.  
   - Injects manual gap rules defined in `MANUAL_GAP_RULES` (currently Station 5 temperature spans) so that “bad but present” readings are treated as proper gaps everywhere else.

2. **`Shortgaps.py`**  
   - Input: cleaned data CSV. Output: `output/Station{id}_filled_shortgaps.csv` plus optional detail log.  
   - SWC columns use monotonic cubic interpolation (PCHIP).  
   - Temperature columns apply time-based interpolation for smoother diurnal signatures.

3. **`Mediumgaps.py`**  
   - Input: short-gap output. Output: `..._filled_mediumgaps.csv` + detail log.  
   - Uses SARIMAX with automatic order selection (24 h seasonality).  
   - Exogenous drivers: `Ppt` by default; for temperature we also attach `Tair`/`Srad` when present.  
   - Training windows are pre-filled locally so temperature models always see a continuous history.

4. **`Longgaps.py`**  
   - Input: medium-gap output. Output: `..._filled_longgaps.csv` + detail log.  
   - XGBoost regression with engineered drivers (lag stats, rolling precipitation, diurnal markers).  
   - Applies start/end drift correction so predictions blend back into real observations.

5. **`VeryLongGaps.py`**  
   - Input: long-gap output. Output: `..._filled_verylonggaps.csv` + detail log.  
   - Chooses the donor station with the highest absolute correlation and ≥1000 overlapping hours, fits a simple linear map, and writes predictions across ≥30-day gaps.  
   - Reports quick MAE/RMSE diagnostics for each parameter.

All outputs share the `output/` directory; each stage overwrites the previous file for that station.

---

## Running the Pipeline

### Recommended: single command orchestrator

From `data-cleanup/imputation_pipeline/` run:

```bash
# all stations, all stages
python imputation_pipeline.py

# single station (e.g., Station 5)
python imputation_pipeline.py --station 5

# dry run to preview commands
python imputation_pipeline.py --dry
```

`imputation_pipeline.py` simply shells through `datacleaning → Shortgaps → Mediumgaps → Longgaps → VeryLongGaps`, reusing the same Python interpreter. If any stage fails, the runner stops immediately so partial outputs are obvious.

### Manual execution (debugging / custom slices)

1. Clean + detect gaps:

   ```bash
   python datacleaning.py --station 4
   ```

2. Fill <24 h gaps:

   ```bash
   python Shortgaps.py --station 4 --param SWC_5 SWC_10
   ```

3. Continue with `Mediumgaps.py`, `Longgaps.py`, `VeryLongGaps.py` as needed. Each script supports `--help` plus `--station` and `--param` overrides (e.g., run temperature parameters alone).

When running by hand, make sure the previous stage’s outputs exist under `output/`; every script reads from there and writes the next suffix.

---

## Outputs & Logs

| Stage | Primary CSV | Detail log |
|-------|-------------|------------|
| datacleaning | `cleaned_data/Station{id}_cleaned_data.csv` | `missing_data/Station{id}_missing_data.csv`|
| Shortgaps | `output/Station{id}_filled_shortgaps.csv` | `output/Station{id}_shortgap_fill_detail.csv` |
| Mediumgaps | `output/Station{id}_filled_mediumgaps.csv` | `output/Station{id}_mediumgap_fill_detail.csv` |
| Longgaps | `output/Station{id}_filled_longgaps.csv` | `output/Station{id}_longgap_fill_detail.csv` |
| VeryLongGaps | `output/Station{id}_filled_verylonggaps.csv` | `output/Station{id}_verylonggap_fill_detail.csv` |

The detail tables list every timestamp that was imputed along with the predicted value and the original gap window. They are useful for auditing and for reintroducing “truth” measurements after model inspection.

---

## Manual Gap Overrides (current)

`datacleaning.py` contains `MANUAL_GAP_RULES`. Today we ship presets for Station 5:

```python
MANUAL_GAP_RULES = {
    5: [
        {"parameters": ["T_20"],
         "start": "2015-01-01 00:00:00",
         "end":   "2016-02-20 12:00:00"},
        {"parameters": ["T_5", "T_10", "T_20", "T_50"],
         "start": "2018-04-14 20:00:00",
         "end":   "2018-05-15 08:00:00"}
    ]
}
```

Why these windows? The visualization notebook flagged two persistent **Station 5** issues:

- **T_20 flatline (2015‑01‑01 → 2016‑02‑20)** – plotted series were locked at ~0 °C for >13 months, so we treat the entire window as missing.
- **Nightly outage (2018‑04‑14 → 2018‑05‑15)** – all soil temperature depths went dark every night (likely power loss). We manually promote that span to a gap so downstream fills use donor stations instead of retaining the zeroed readings.

Feel free to extend this structure when new persistent anomalies surface. The injection happens *before* we write the missing-summary CSV, so every downstream stage will automatically treat the span as a real gap.

---

## Tips

- **Reinserting real observations**: after VeryLongGaps, reapply any trusted measurements by masking `filled_verylonggaps` with the original cleaned data (keep truth where it exists, keep model output elsewhere).
- **Visualization notebooks**: Check the notebook: `data_visualization/Dynamic_Data_Visualization.ipynb` 
- **Storage**: keep an eye on `output/` size; each station/stage writes a full hourly history. It’s safe to prune intermediate stages once you confirm the full run.

## Method Error Matrix

The following matrix summarizes which modeling family we rely on for each gap length category (mirrors the shared Google Sheet).

| Parameters \ Gaps | Short Gaps (1–24 hr) | Med. Gaps (1–7 days) | Long Gaps (1 week–1 month) | Very Long Gaps (≥30 days) |
|--------------------|----------------------|----------------------|----------------------------|---------------------------|
| Methods            | Interpolation, kNN   | ARIMA, SARIMA        | Baseline (seasonal-hour mean + slow linear trend), Random Forest, LightGBM, XGBoost | Linear regression donor-station infilling |

Refer to the source sheet (`https://docs.google.com/spreadsheets/d/1OouqvV3Te1l-xHxy8NfNVA0TpEiFaFh0Agav3eePHf8/edit#gid=0`) for additional notes or historical error statistics per parameter. For detailed notebook walkthroughs, switch to branch `zun-cao` and open `Model_2025/Short Gaps.ipynb`, `Model_2025/SARIMA.ipynb`, `Model_2025/Long Gaps.ipynb`, and `Model_2025/Very_Long_Gaps.ipynb`.

### Known unresolved sensors

- **Station 4 SWC_50 / T_50** – every record in the raw files is already `NaN`. No donor-based patch can fix an all-null column, so these stay missing through the pipeline. Any downstream analysis should treat Station 4 depth-50 readings as unavailable.

With these pieces in place you can regenerate Station 1–6 with consistent soil moisture and temperature fills in a single command and audit every value that was touched.

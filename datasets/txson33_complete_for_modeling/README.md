# Complete TxSON33 Dataset for Modeling

## Overview

The original TxSON imputation pipeline fills valid gaps within each sensor's
observed coverage. It intentionally preserves structural missingness when a
sensor was never installed, a timestamp falls outside source coverage, or a
station has no dedicated MET measurements.

Modeling workflows need a complete feature matrix across all 33 stations. This
dataset starts from the base imputation output and applies a separate,
modeling-only completion step to the remaining missing values.

No existing finite value in the base output is changed. Additional values are
labeled `modeling_completion` so they remain distinguishable from observations
and values filled by the original imputation pipeline.

## Quick Start

The repository includes `TxSON33_modeling_complete.parquet`; use it for Python
analysis:

```python
import pandas as pd

df = pd.read_parquet("TxSON33_modeling_complete.parquet")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
```

The approximately 916 MB CSV is not tracked in Git. Generate a local CSV from
the included Parquet file when a portable format is needed:

```python
import pandas as pd

df = pd.read_parquet("TxSON33_modeling_complete.parquet")
df.to_csv("TxSON33_modeling_complete.csv", index=False)
```

## Dataset Contents

- Stations: **33**
- Rows: **2,720,775**
- Overall date range: **2014-09-02 15:00:00** to **2026-02-24 15:00:00**
- Required measurement variables: **14**
- Missing required measurement values: **0**
- Unique key: `Station` + `Timestamp`

Measurement columns:

- Soil moisture: `SWC_5`, `SWC_10`, `SWC_20`, `SWC_50`
- Soil temperature: `T_5`, `T_10`, `T_20`, `T_50`
- Weather: `Tair`, `RH`, `Srad`, `Wind speed`, `Wind direction`, `Ppt`

## How Remaining Missing Values Were Completed

The modeling-only step fills only values that were still missing in the base
imputation output.

- Soil variables use deterministic pooled gradient-boosting models with other
  Soil depths, `Ppt`, station identity, hour, and day-of-year features.
- MET variables use concurrent network information from the six dedicated-MET
  stations, followed by time-based climatology when concurrent data are absent.
- Wind direction is treated circularly so angles near north remain close.
- `Ppt` was already complete and is unchanged.

## Origin Columns

Every measurement has a matching `*_Origin` column:

- `observed`: unchanged from a finite source observation.
- `pipeline_imputed`: filled or replaced by the original imputation pipeline.
- `modeling_completion`: added only to complete the modeling matrix.

`Any_Soil_Imputed` and `Any_MET_Imputed` retain the original pipeline's row-level
flags. `Any_Modeling_Completion` identifies rows where the modeling-only step
added at least one value.

For evaluation, filter on the prediction target's own `*_Origin` column rather
than using a row-level flag.

## Data Coverage

Dedicated non-Ppt MET measurements exist at `CB01`, `CB04`, `CB06`, `FD02`,
`FD03`, and `WC05`. The other 27 stations use regional estimates for non-Ppt
MET variables.

`CB07`, `CB26`, `FD03`, `FD18`, `FD21`, and `FD24` originally lacked both
`SWC_50` and `T_50`.

- Highest original-data coverage: `FD02, CB04, CB06, WC05, FD03`.
- Lowest original-data coverage: `CB15, CB20, FD12, CB19, FD24`.

See `data_quality_summary.csv` for observed, pipeline-imputed, and
modeling-completion counts and percentages for every station-variable pair.

## Key Notes

- `modeling_completion` values are synthetic estimates and should not normally be
  used as validation or test ground truth for that variable.
- Only six stations have dedicated non-Ppt MET; the other 27 use regional
  estimates.
- `RH` and `Wind direction` have higher completion uncertainty than the other
  MET variables.

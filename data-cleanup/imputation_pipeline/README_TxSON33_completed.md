# TxSON 33-Station Cleanup: Completed Work

This note documents only the completed and verified work for the new 33-station dataset:

```text
datasets/TxSON_data_2026-02-24/
```

## Completed Pipeline

```text
Raw .dat files
    -> datacleaning.py
    -> cleaned_data / missing_data / raw_merged_data
    -> Shortgaps.py
    -> output/*_filled_shortgaps.csv
    -> summaries and visualizations
```

## Stage 0: Data Cleaning

Script:

```text
data-cleanup/imputation_pipeline/datacleaning.py
```

What it now supports:

- New station IDs such as `CB01`, `FD08`, `WC05`
- Old station IDs such as `1`, `2`, ..., `6`
- New soil files named `{site}.dat`
- New MET files named `{site}_met.dat`
- Old files named `SM_{id}.dat` and `MET_{id}.dat`
- Soil files with citation/header text before the `Date` row
- Campbell TOA5 MET files
- Stations with no MET file
- Duplicate timestamps
- Sub-hourly records, aggregated to hourly data

Important cleaning logic:

- Final output is reindexed to a complete hourly timeline.
- Invalid physical values are replaced with `NaN`.
- Missing summaries are generated from the full hourly timeline.
- Short missing summaries with no rows still keep the standard CSV header.

Run one station:

```powershell
cd data-cleanup/imputation_pipeline

python datacleaning.py --station CB01 `
  --soil-base-dir ../../datasets/TxSON_data_2026-02-24 `
  --met-base-dir ../../datasets/TxSON_data_2026-02-24
```

Main outputs:

```text
cleaned_data/StationCB01_cleaned_data.csv
missing_data/StationCB01_missing_data.csv
raw_merged_data/raw_merged_station_CB01.csv
```

## Short-Gap Filling

Script:

```text
data-cleanup/imputation_pipeline/Shortgaps.py
```

What it now supports:

- Site-code station IDs such as `CB01`
- Auto-discovery of `Station{site}_cleaned_data.csv`
- Missing columns are skipped safely

Short-gap rules:

- Gaps shorter than 24 hours are filled.
- `SWC_*` uses PCHIP interpolation.
- `T_*` and `Tair` use time interpolation.
- `Ppt` short gaps are filled with `0`.
- `Wind direction` uses vector interpolation.

Run all discovered 33 stations:

```powershell
cd data-cleanup/imputation_pipeline

$sites = Get-ChildItem ..\..\datasets\TxSON_data_2026-02-24 -Filter *.dat |
  Where-Object { $_.Name -notlike '*_met.dat' } |
  Sort-Object BaseName |
  ForEach-Object { $_.BaseName }

python Shortgaps.py --station $sites
```

Main outputs:

```text
output/StationCB01_filled_shortgaps.csv
output/StationCB01_shortgap_fill_detail.csv
```

Stations with no short gaps still get a `filled_shortgaps.csv` file.

## Validation Outputs

Generated validation files:

```text
data-cleanup/imputation_pipeline/stage0_summary.csv
data-cleanup/imputation_pipeline/shortgaps_summary.csv
```

Validation checks completed:

- 33/33 stations produced cleaned files.
- 33/33 stations produced missing summary files.
- 33/33 stations produced raw merged files.
- 33/33 stations produced short-gap output files.
- No duplicate datetime indexes.
- No non-hourly steps after cleaning.
- No shape mismatch after short-gap filling.
- NaN counts decreased or stayed unchanged after short-gap filling.

## Visualizations

Generated overview script:

```text
data_visualization/visualize_txson_33_gaps.py
```

Generated reports:

```text
data_visualization/txson_33_gap_reports/txson_33_gap_overview.html
data_visualization/txson_33_gap_reports/txson_33_gap_details.csv
data_visualization/txson_33_gap_reports/txson_33_station_missing_metadata.csv
```

Generate or refresh reports:

```powershell
python data_visualization\visualize_txson_33_gaps.py
```

Generate a station-level missing timeline:

```powershell
python data_visualization\visualize_txson_33_gaps.py --station CB01
```

Example output:

```text
data_visualization/txson_33_gap_reports/txson_CB01_missing_timeline.html
```

## Dynamic Notebook

New notebook for the 33-station data:

```text
data_visualization/Dynamic_Data_Visualization_TxSON33.ipynb
```

Supporting module:

```text
data_visualization/txson33_dynamic_visualization.py
```

Controls:

```text
Plot Type
Station
Year
Month
MET Var
Time Type
```

Data source priority:

```text
output/Station{site}_filled_shortgaps.csv
cleaned_data/Station{site}_cleaned_data.csv
```

Notebook behavior:

- Station list is discovered automatically.
- Year options update by station.
- Month options update by station and year.
- Plotly lines break at `NaN` gaps.
- Missing timestamps are marked in red.

## Not Included Yet

The following stages are not documented here because they have not been completed for the 33-station dataset:

```text
Mediumgaps.py
Longgaps.py
VeryLongGaps.py
```

# TxSON Data Visualization

## Zun Cao

## Current 33-Station Dashboard

Use [`Dynamic_Data_Visualization_TxSON33.ipynb`](Dynamic_Data_Visualization_TxSON33.ipynb)
for the current TxSON 33-station results. Its implementation is kept in
[`txson33_dynamic_visualization.py`](txson33_dynamic_visualization.py).

For each station, the dashboard combines these two final products by timestamp:

```text
data-cleanup/imputation_pipeline/output/Station{site}_filled_final.csv
data-cleanup/imputation_pipeline/met_output/Station{site}_met_filled_complete.csv
```

The soil file supplies soil moisture and soil temperature. The MET complete
file supplies `Ppt`, `Tair`, `RH`, `Srad`, `Wind speed`, and `Wind direction`.
The merge happens only in memory for plotting; the notebook does not create or
modify a delivery CSV. The loader requires paired final Soil and final MET files
and fails clearly when either product is unavailable; it never substitutes an
earlier pipeline stage.

From the repository root:

```bash
source .venv/bin/activate
jupyter lab data_visualization/Dynamic_Data_Visualization_TxSON33.ipynb
```

The dashboard supports station, year, month, total-period, soil-depth, and MET
parameter selection. Red ticks identify any NaN timestamps still present in
the selected source.

## Stage 0 Gap Report

[`visualize_txson_33_gaps.py`](visualize_txson_33_gaps.py) summarizes the gaps
found by Stage 0 and can optionally show the soil short-gap output. It is a
diagnostic inventory, not a viewer for the final merged delivery.

```bash
python data_visualization/visualize_txson_33_gaps.py
python data_visualization/visualize_txson_33_gaps.py --station CB01
```

Generated HTML and CSV reports are written to
`data_visualization/txson_33_gap_reports/`.

## Legacy Six-Station Tools

The following files target the earlier numeric station dataset under
`datasets/Revised_Final_Data` or `datasets/New_Revised_Final_Data`. They are
retained for reproducibility but are not part of the TxSON 33-station pipeline:

```text
Dynamic_Data_Visualization.ipynb
script_plot_swc_yearly.py
script_plot_swc_monthly.py
script_plot_swc_violin.py
script_plot_ppt_monthly.py
```

Install the shared environment from the pipeline requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r data-cleanup/imputation_pipeline/requirements.txt
```

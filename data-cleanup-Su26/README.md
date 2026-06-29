![pipeline](./images/TxSON-pipeline.png)

# TxSON Data Cleanup

Scripts and notebooks to ingest (read → prewash → clean) the raw TxSON network `.dat` files.

## Prewash_df.py
- The full prewash pipeline in one standalone script. Only requires Pandas and datetime to run.

- **CLI:** `python prewash_df.py <input.dat> <output.csv>` (output.csv defaults to working directory)


## Notebooks

- **DST_check** — checks whether the TxSON network observes daylight savings, and the data quirks found.
- **duplicate_check** — builds the tools that find all three duplicate cases; set the data path at the top and run all to regenerate the report (see `duplicate_report/README.md`).
- **data_visuals** — interactive Plotly explorer for the prewashed data; overlay variables and spot gaps (shown as red blocks).

## Folders

- **scripts** — read + prewash + clean the raw TxSON `.dat` files.
- **duplicate_report** — the dropped + kept rows for each of the three duplicate cases.

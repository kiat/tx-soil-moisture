# TxSON CNN Soil-Moisture Forecasting

A clean, reproducible reorganization of the CNN soil-moisture forecasting study
on the **Texas Soil Observation Network (TxSON)** dataset. The target is soil
water content at 20 cm, `SWC_20`; a 1D CNN forecasts it from a 24-hour input
window at horizons of 24–168 hours.

**Research question.** *Which exogenous (external) features most improve CNN
soil-moisture forecasting — precipitation, lagged/cumulative rainfall, air
temperature, humidity, seasonality — relative to using historical soil moisture
alone?*

## Directory layout

```
soil_moisture_research/
├── README.md              this file
├── requirements.txt       dependencies (core = no GPU; torch for training)
├── main.py                CLI experiment runner (shell / overnight)
├── src/                   reusable pipeline (imported by notebooks & main.py)
│   ├── config.py          paths, stations, splits, feature groups
│   ├── features.py        causal feature engineering + lag correlation
│   ├── data_processing.py loading, windowing, held-out folds
│   ├── models.py          SmallCNN / LargerCNN / ConfigurableCNN
│   ├── training.py        seeded training loop + early stopping
│   └── evaluation.py      metrics, persistence baseline, fold harness
├── notebooks/             eight small, single-purpose notebooks (see below)
├── results/               saved experiment CSVs (150 CNN runs + baselines)
├── figures/               generated figures (PNG)
├── report/                LaTeX report (main.tex) + report figures (PDF)
└── archive/               backup of the original monolithic notebook
```

## Notebooks (run in order; each is self-contained)

**Which notebooks train a CNN vs. analyze saved results?** Only **`02`** trains a
CNN inside the notebook (and carries real Colab-GPU outputs). The large experiment
grids were trained once on 2026-11-17 and saved to `results/`; notebooks `03`,
`05`, `06`, `07` **load those saved results** and analyze/plot them (they also show
the exact `src`/`main.py` code to regenerate them on a GPU). Notebooks `01` and
`04` **compute fresh** from the raw dataset (feature engineering, lag correlations);
`08` regenerates figures from the saved CSVs.

| Notebook | Purpose | Trains a CNN? | Results shown |
|---|---|---|---|
| `01_data_preprocessing` | Load dataset, schema, causal feature engineering | no | 🔄 computed fresh |
| `02_cnn_baseline` | Train LargerCNN on one held-out fold (smallest full example) | **yes (in-notebook, GPU)** | 🔄 fresh GPU run (2026-11-19) |
| `03_feature_comparison` | Configs A–F × 5 station-years × 5 horizons (150 runs) | no (loads; GPU loop shown) | ⬇ Nov-17 CSV |
| `04_precipitation_lag_analysis` | Rain→moisture lag; 5 cm vs 20 cm depth | no | 🔄 computed fresh |
| `05_horizon_experiments` | RMSE vs forecast horizon vs persistence | no (loads) | ⬇ Nov-17 CSV |
| `06_hyperparameter_tuning` | Kernel / depth / filters / dropout / lr sweep | no (loads; GPU snippet shown) | ⬇ Nov-17 CSV |
| `07_feature_importance` | **Exogenous feature interpretation (main result)** | no (loads) | ⬇ derived from Nov-17 |
| `08_results_visualization` | Regenerate all report figures from CSVs | no | ⬇ Nov-17 CSVs |

Every notebook starts with the same portable **setup cell** that locates the
project root (the folder containing `src/`) and adds it to the path — it works
unchanged from a local checkout, a Google-Drive upload, or a fresh Colab session.
The torch-free notebooks run on any machine with the core dependencies; `02` (and
the regeneration loops in `03`/`06`) need PyTorch and are best run on Colab/GPU.

## Running in Google Colab (fresh session)

The project is designed to run from a clean Colab runtime with no path edits:

1. **Put the whole `soil_moisture_research/` folder in Google Drive** (keep the
   folder structure — `src/`, `results/`, `notebooks/` must stay together).
2. In Colab, open any notebook from `notebooks/` (e.g. drag it in, or
   `File → Open notebook → Google Drive`). For `02`, set **Runtime → Change
   runtime type → GPU**.
3. If the notebook is opened *outside* the folder, mount Drive first so the setup
   cell can find the project:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   # then make sure the folder is at /content/drive/MyDrive/soil_moisture_research
   ```
4. Run the cells top to bottom. The **dataset auto-downloads** (~85 MB) from the
   public GitHub mirror on first use — no manual upload needed. The committed
   `results/*.csv` are read directly from the uploaded folder.

> Verified on a fresh Colab runtime (2026-11-19): the `src/` package imports, the
> dataset auto-downloads, notebook `02` trains one fold on GPU and reproduces the
> saved RMSE, and the relative `results/` reads resolve from the project root.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The dataset `TxSON33_modeling_complete.parquet` (~85 MB) is found automatically
if it sits in `data/`, in the known local checkout, or on Colab Drive. Override
with `export TXSON_PARQUET=/path/to/TxSON33_modeling_complete.parquet`. If it is
missing, `src.data_processing.load_dataframe` downloads it to `data/`.

## Reproducing each experiment

> The committed `results/nov17_results.csv` (150 CNN runs + 25 persistence rows)
> comes from the original Nov-17 Colab session (archived pipeline in `archive/`).
> `main.py` re-runs the *same pipeline* and writes its **own** fresh output to
> `results/experiments.csv` / `results/loss_histories.csv`, so reproducing never
> overwrites the original results.

**Feature comparison / horizons (the 150-run grid).** `--horizon` takes **one**
value per call, so loop over horizons (GPU recommended):

```bash
for H in 24 48 72 96 168; do
  python main.py --horizon $H \
    --features swc,existing_rain,lagged_rain,cumulative_rain,temperature,combined_met
done
```

Re-running **skips** completed `(features, station, year, horizon)` rows, so an
interrupted job resumes. Overnight, detached:

```bash
nohup bash -c 'for H in 24 48 72 96 168; do python -u main.py --horizon $H --features swc; done' \
  > experiment.log 2>&1 < /dev/null &
tail -f experiment.log
```

**Hyperparameter sweep** (one variable at a time, 72 h, FD11-2023):

```bash
python main.py --horizon 72 --arch configurable --kernel-size 7 \
  --features swc --stations FD11 --test-years 2023
```

**Lag analysis, feature ranking, figures** (no GPU) — run notebooks 04, 07, 08,
or:

```bash
python notebooks/make_figures.py      # regenerate every figure from results/
```

## Method (summary)

- **Held-out station-year evaluation.** Test one `(station, year)`; validate on
  2019 of the *other* stations; train on everything else. The scaler
  (`MinMaxScaler`) is fit on training data only, and metrics are computed in
  original SWC units after inverse-transform.
- **No future leakage.** All rainfall accumulations are trailing, lagged features
  use past shifts, meteorology is forward-filled — never backward-filled. Windows
  never cross a year/station boundary.
- **Model selection** uses validation loss only; the held-out test set is touched
  once, for the final metric.
- **Reproducibility.** Fixed seed (42) for Python/NumPy/torch; results logged
  incrementally; the pipeline is identical across notebooks and `main.py`.

## Key results

The complete numerical findings are in `results/` and the report
(`report/main.tex`). Headline: across 5 held-out station-years × 5 horizons,
**no exogenous feature consistently improves RMSE over soil-moisture-only** — the
historical `SWC_20` signal dominates. The only clear positive is cumulative
rainfall at the driest station (FD02). See `notebooks/07_feature_importance`.

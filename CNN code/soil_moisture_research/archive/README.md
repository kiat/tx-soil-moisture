# archive/

Backups and superseded artifacts. Nothing here is needed to run the project.

- **`cnn_experiments_original_monolithic.ipynb`** — backup of the original
  single large notebook (the committed `../../CNN code/cnn_experiments.ipynb`,
  ~190 cells) before it was refactored into `src/` modules and the eight small
  notebooks in `../notebooks/`. Kept verbatim as a record.

### Note on the November 17 work
The live Colab notebook `cnn_experiments_(5)` (≈218 cells) additionally contains
the November 17 section (P1–P11: precipitation-lag analysis, horizon/feature
comparison, hyperparameter sweep, loss diagnostics). That section's **code** has
been refactored into `../src/` and `../notebooks/04`–`08`, and its **results**
(the 150 CNN runs + 25 persistence baselines, lag correlations, loss histories)
are saved under `../results/`. The original monolithic notebook was not deleted.

# data/

The dataset `TxSON33_modeling_complete.parquet` (~85 MB) is **not committed**
(it is large and lives upstream). `src/data_processing.load_dataframe` finds it
automatically if present in this folder, in the local checkout, or on Colab
Drive, and otherwise downloads it here. To point at an existing copy:

```bash
export TXSON_PARQUET=/path/to/TxSON33_modeling_complete.parquet
```

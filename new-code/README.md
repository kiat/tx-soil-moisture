# Soil Moisture Prediction

This repository trains and evaluates deep learning models for soil moisture prediction using time series data. The pipeline supports **custom configurations via command-line arguments**.

## Installation

Ensure you have **Python 3.8+** installed. Dependencies are managed via `pyproject.toml`. To install them, run:

```bash
pip install .
```

or, if using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install .
```

## Running the Experiment

Run the experiment using:

```bash
python3 src/main.py
```

This will use default settings from `config.yaml`.

### Overriding Parameters via Command-Line Arguments

Command-line arguments can override default parameters without modifying the config file.

#### Example: Specify Window Sizes, Epochs, and Batch Size

```bash
python3 src/main.py --window_sizes 24 48 --epochs 5 --batch_size 64
```

#### Example: Specify Specific Stations and SWC Variables

```bash
python3 src/main.py --station_list Station1 Station2 --swc_list SWC_5 SWC_10
```

#### Example: Specify Forecasting Offset

```bash
python3 src/main.py --offset 12
```

## Command-Line Arguments

| Argument         | Description                                         | Example Usage                           |
|-----------------|-----------------------------------------------------|-----------------------------------------|
| `--window_sizes` | List of window sizes to test                       | `--window_sizes 24 48 72`               |
| `--offset`       | Forecasting offset (hours)                         | `--offset 24`                           |
| `--swc_list`     | List of SWC columns to predict                     | `--swc_list SWC_5 SWC_20`               |
| `--station_list` | List of stations to use                            | `--station_list Station1 Station2`      |
| `--epochs`       | Number of training epochs                          | `--epochs 10`                           |
| `--batch_size`   | Batch size for model training                      | `--batch_size 64`                       |

## Output Structure

Each run saves results in a timestamped folder inside `results/`:

```
results/
└── run_20250220_135645/
    ├── Simple_LSTM_checkpoint.keras
    ├── Simple_LSTM_loss_plot.png
    ├── Simple_LSTM_predictions.png
    ├── model_comparison_results.csv
    ├── experiment_parameters.yaml  <-- Contains all experiment settings
```

## Example: Full Run with Custom Parameters

```bash
python3 src/main.py --window_sizes 24 48 72 --offset 12 --epochs 5 --batch_size 64 --station_list Station1 --swc_list SWC_5 SWC_10
```

This will:
- Train models using **window sizes 24, 48, 72**.
- Predict with an **offset of 12 hours**.
- Train for **5 epochs** with a **batch size of 64**.
- Use **Station1** for prediction on **SWC_5** and **SWC_10**.


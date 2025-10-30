# Time Series Prediction with Deep Learning Models

This repository contains scripts to preprocess time series data, train multiple deep learning models (LSTM, BiLSTM, RNN, CNN) for time series forecasting, and evaluate their performance. To learn how the code works from start to finish, a good place to start would be `main.py`, which has comments throughout the code explaining what each section does, as well as what methods are used.

## Prerequisites

This project now uses PyTorch. Install dependencies with either Conda (`environment.yml`) or pip (`requirements.txt`). Core packages:

```txt
torch
torchvision
pandas
numpy
scikit-learn
scipy
fastparquet
matplotlib
seaborn
tensorboard
tqdm
jupyter
```

Optional GPU: install the CUDA build of PyTorch from pytorch.org per your driver/version.

## File Descriptions

-   `main.py`: Entry point. Orchestrates data prep, training, evaluation, and saving.
-   `models/`: Modular PyTorch models and a registry system (see Model Zoo below).
-   `models.py`: Backward-compatibility shim that re-exports from `models/`.
-   `utils/`: Training/evaluation/data utilities (`Trainer`, `Evaluator`, `prepare_dataloaders`, callbacks).
-   `core/data_helpers.py`: Data I/O, feature engineering, normalization, and dataset creation.
-   `core/evaluation_helpers.py`: CSV writers for metrics and training history.
-   `Revised_Final_Data/`: Cleaned, timestamp-aligned station CSVs.
-   `saved_models_pytorch/`: Directory where trained `.pth` model checkpoints are saved.
-   `results/`: Output CSVs for metrics and loss history.
-   `requirements.txt` and `environment.yml`: Dependency manifests.

## Repository Structure

```
src-pytorch/
├─ main.py
├─ models/
│  ├─ __init__.py
│  ├─ baseline.py
│  ├─ lstm.py
│  ├─ rnn_cnn.py
│  ├─ attention.py
│  ├─ other.py
│  └─ registry.py
├─ utils/
│  ├─ __init__.py
│  ├─ callbacks.py
│  ├─ trainer.py
│  ├─ evaluator.py
│  └─ data_utils.py
├─ core/
│  ├─ data_helpers.py
│  └─ evaluation_helpers.py
├─ results/
├─ saved_models_pytorch/
├─ Revised_Final_Data/
├─ run_experiments.sh
├─ requirements.txt
└─ environment.yml
```

## Running the Project

### Step 1: Train Models

Run the training script (defaults shown on the right):

```bash
python3 main.py \
    --window_size 48 \
    --offset 48 \
    --epochs 30 \
    --patience 10 \
    --batch_size 32 \
    --predictors "SWC_20,T_20,Ppt,Wx,Wy,Srad,DayCos,DaySin,MonthCos,MonthSin" \
    --predict_features "SWC_20" \
    --model_names "bilstm"
```

Parameters:

### Available Arguments

| Argument                 | Description                            | Default                                                      |
| ------------------------ | -------------------------------------- | ------------------------------------------------------------ |
| `--window_size`          | Input sequence length (time steps)     | `48`                                                         |
| `--label_width`          | Output width (prediction steps)        | `1`                                                          |
| `--training_stride`      | Sliding window stride for training     | `8`                                                          |
| `--validation_stride`    | Sliding window stride for validation   | `8`                                                          |
| `--offset`               | Forecast horizon (time steps ahead)    | `48`                                                         |
| `--epochs`               | Max training epochs                    | `30`                                                         |
| `--patience`             | Early stopping patience                | `10`                                                         |
| `--batch_size`           | Batch size                             | `32`                                                         |
| `--predictors`           | Comma-separated input features         | `SWC_20,T_20,Ppt,Wx,Wy,Srad,DayCos,DaySin,MonthCos,MonthSin` |
| `--features`             | Deprecated; same as predictors         | `SWC_20,T_20,Ppt,Tair,Wx,Wy`                                 |
| `--predict_features`     | Targets to predict                     | `SWC_20`                                                     |
| `--daily_average_output` | Output daily averages (flag)           | `False`                                                      |
| `--model_names`          | Comma-separated model names (registry) | `bilstm`                                                     |

---

### Step 2: Run Experiments with Different Offsets

To train models with multiple offsets, execute:

```bash
bash run_experiments.sh
```

This script tests offsets of 24, 72, and 168 while keeping other parameters constant.

### Step 3: Evaluate Model Performance

After training, results are saved in CSV files:

-   `results_<model>_ws<window_size>_offset<offset>_<features>.csv`: Stores model evaluation metrics, including:

    -   `r2_score`: Measures how well the predictions match actual values.
    -   `mean_squared_error (MSE)`: Indicates average squared error.
    -   `mean_absolute_error (MAE)`: Shows absolute difference between predicted and actual values.
    -   `mean_absolute_percentage_error (MAPE)`: Expresses error as a percentage.
    -   `symmetric mean absolute percentage error (SMAPE)`: Provides a symmetric error measure.
    -   `relative squared error (RSE)`: Evaluates model performance in comparison to a naive baseline.
    -   `correlation coefficient (CORR)`: Measures correlation between predictions and actual values.

-   `loss_history_<model>_ws<window_size>_offset<offset>_<features>.csv`: Stores loss history per epoch, including:
    -   `loss`: Training loss.
    -   `validation loss`: Loss on the validation set.
    -   Epoch-wise progression of loss values.

## Model Outputs

Trained models are saved in `saved_models_pytorch/` as PyTorch checkpoints (`.pth`). Filenames include model name, window size, offset, and features.

# Example Command

```
python3 main.py --window_size 48 --offset 48 --epochs 30 --patience 10 --predictors "SWC_10,Ppt" --model_names bilstm
```

## Model Zoo and Registry

Models are selected via the registry (case-insensitive). Valid names and aliases:

| Family         | Class                   | Registry names (aliases) |
| -------------- | ----------------------- | ------------------------ |
| Baselines      | `Baseline`              | `baseline`               |
| Baselines      | `MovingAverageBaseline` | `movingaverage`          |
| RNNs           | `LSTMModel`             | `lstm`                   |
| RNNs           | `BiLSTMModel`           | `bilstm`                 |
| RNNs           | `RNNModel`              | `rnn`                    |
| CNNs           | `CNNModel`              | `cnn`                    |
| Attention      | `AttentionLSTM`         | `attentionlstm`          |
| Attention      | `AttentionOnly`         | `attentiononly`          |
| Attention      | `MultiHeadLSTM`         | `multiheadlstm`          |
| Transformer    | `TransformerModel`      | `transformer`            |
| Autoregressive | `AR`                    | `ar`, `autoregressive`   |
| Interpretable  | `ILSTM_Soil`            | `ilstmsoil`              |

Notes:

-   ILSTM_Soil is instantiated automatically with `time_steps` and `num_features` from the data; other models use `input_dim`.
-   Backward-compatibility: importing `models.py` still works, but new code should import from the `models/` package.

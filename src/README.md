# Time Series Prediction with Deep Learning Models

This repository contains scripts to preprocess time series data, train multiple deep learning models (LSTM, BiLSTM, RNN, CNN) for time series forecasting, and evaluate their performance.

## Prerequisites

Before running the project, ensure you have the following dependencies installed. You can save the following in a `requirements.txt` file and install using `pip install -r requirements.txt`:

```txt
pandas
numpy
tensorflow
scikit-learn
fastparquet
argparse
stats
keras-attention
keras_self_attention
```

Alternatively, install manually using:

```bash
pip install pandas numpy tensorflow scikit-learn fastparquet argparse stats 
```

## File Descriptions

- `main.py`: Main script to train different models and evaluate performance.
- `preprocess_data.py`: Script for data preprocessing, including loading CSV files, engineering features, and saving the data as Parquet.
- `run_experiments.sh`: Bash script to run experiments with different offsets.

## Running the Project

### Step 1: Train Models

Run the training script with default parameters:

```bash
python3 main.py --window_size 168 --offset 24 --epochs 10 --patience 3 --features "SWC_20,T_20,Ppt,Tair,Wx,Wy"
```

Parameters:

### Available Arguments

| Argument        | Description                                 | Default                                                       |
| --------------- | ------------------------------------------- | ------------------------------------------------------------- |
| `--window_size` | Size of input window (in time steps)        | `168`                                                         |
| `--offset`      | Forecasting horizon (in time steps)         | `24`                                                          |
| `--epochs`      | Number of training epochs                   | `10`                                                          |
| `--patience`    | Early stopping patience                     | `3`                                                           |
| `--features`    | Comma-separated list of input features      | `"SWC_20,T_20,Ppt,Tair,Wx,Wy"`                                |
| `--model_names` | Comma-separated model names to train        | `"LSTM,BiLSTM,RNN,CNN,AttentionLSTM,Autoregressive,Baseline"` |
| `--visualize`   | Plot data splits instead of training models | *(optional flag)*                                             |

---


### Step 2: Run Experiments with Different Offsets

To train models with multiple offsets, execute:

```bash
bash run_experiments.sh
```

This script tests offsets of 24, 72, and 168 while keeping other parameters constant.

### Step 3: Evaluate Model Performance

After training, results are saved in CSV files:

- `results_<model>_ws<window_size>_offset<offset>_<features>.csv`: Stores model evaluation metrics, including:
  - `r2_score`: Measures how well the predictions match actual values.
  - `mean_squared_error (MSE)`: Indicates average squared error.
  - `mean_absolute_error (MAE)`: Shows absolute difference between predicted and actual values.
  - `mean_absolute_percentage_error (MAPE)`: Expresses error as a percentage.
  - `symmetric mean absolute percentage error (SMAPE)`: Provides a symmetric error measure.
  - `relative squared error (RSE)`: Evaluates model performance in comparison to a naive baseline.
  - `correlation coefficient (CORR)`: Measures correlation between predictions and actual values.

- `loss_history_<model>_ws<window_size>_offset<offset>_<features>.csv`: Stores loss history per epoch, including:
  - `loss`: Training loss.
  - `validation loss`: Loss on the validation set.
  - Epoch-wise progression of loss values.

## Model Outputs

Trained models are saved in the `models/` directory as `.keras` files. The model filenames follow this pattern:

```plaintext
<model_name>_<station_name>.keras
```

For example:

```plaintext
LSTM_Station1.keras
BiLSTM_Station3.keras
CNN_Station5.keras
```

These files contain the trained deep learning models, which can be loaded for inference.

## Data Visualization

To visualize the results, use the `visualize.py` script. This script generates plots for the training and validation loss over epochs, as well as the predicted vs. actual values, for 
```bash
python3 main.py --features "SWC_10" --visualize
```
This will create plots in the `results/visualizations` directory.

## Notes

- Ensure data files (`StationX_Revised_Final_Data.csv`) are available before preprocessing.
- Modify `run_experiments.sh` to customize offsets, features, or model parameters.



# Example Command 

```
python3 main.py --window_size 6 --offset 2 --epochs 1 --patience 1 --features "SWC_10,Ppt" --model_names "LSTM"
```


# Clean up 

```
make clean 
```






















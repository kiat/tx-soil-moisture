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

### Step 1: Preprocess Data

Before training models, preprocess raw CSV files into Parquet format:

```bash
python3 preprocess_data.py
```

This script will:
- Read CSV files (`StationX_Revised_Final_Data.csv`).
- Apply feature engineering.
- Save processed data as Parquet files.

### Step 2: Train Models

Run the training script with default parameters:

```bash
python3 main.py --window_size 168 --offset 24 --epochs 10 --patience 3 --features "SWC_20,T_20,Ppt,Tair,Wx,Wy"
```

Parameters:
- `--window_size`: Number of time steps used for training (default: 168).
- `--offset`: How far ahead to predict (default: 24).
- `--epochs`: Number of training epochs (default: 10).
- `--patience`: Early stopping patience (default: 3).
- `--features`: Comma-separated list of features to use.
-  `--model_names`: "LSTM"

### Step 3: Run Experiments with Different Offsets

To train models with multiple offsets, execute:

```bash
bash run_experiments.sh
```

This script tests offsets of 24, 72, and 168 while keeping other parameters constant.

### Step 4: Evaluate Model Performance

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




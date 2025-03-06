Time Series Prediction with Deep Learning Models

This repository contains scripts to preprocess time series data, train multiple deep learning models (LSTM, BiLSTM, RNN, CNN) for time series forecasting, and evaluate their performance.

Prerequisites

Before running the project, ensure you have the following dependencies installed:
```pip install pandas numpy tensorflow scikit-learn fastparquet```

File Descriptions

main.py: Main script to train different models and evaluate performance.

preprocess_data.py: Script for data preprocessing, including loading CSV files, engineering features, and saving the data as Parquet.

run_experiments.sh: Bash script to run experiments with different offsets.

Running the Project

Step 1: Preprocess Data

Before training models, preprocess raw CSV files into Parquet format:
```python3 preprocess_data.py```
This script will:

Read CSV files (StationX_Revised_Final_Data.csv).

Apply feature engineering.

Save processed data as Parquet files.

Step 2: Train Models

Run the training script with default parameters:
```python3 main.py --window_size 168 --offset 24 --epochs 10 --patience 3 --features "SWC_20,T_20,Ppt,Tair,Wx,Wy"```
Parameters:

```--window_size:`` Number of time steps used for training (default: 168).

```--offset:``` How far ahead to predict (default: 24).

```--epochs:`` Number of training epochs (default: 10).

```--patience:``` Early stopping patience (default: 3).

```--features:``` Comma-separated list of features to use.

Step 3: Run Experiments with Different Offsets

To train models with multiple offsets, execute:

```bash run_experiments.sh```

This script tests offsets of 24, 72, and 168 while keeping other parameters constant.

Step 4: Evaluate Model Performance

After training, results are saved in CSV files:

````results_<model>_ws<window_size>_offset<offset>_<features>.csv```: Stores model evaluation metrics.

```loss_history_<model>_ws<window_size>_offset<offset>_<features>.csv```: Stores loss history for each epoch.

Model Outputs

Trained models are saved in the models/ directory as .keras files.

Notes

Ensure data files (StationX_Revised_Final_Data.csv) are available before preprocessing.

Modify run_experiments.sh to customize offsets, features, or model parameters.

License

This project is for educational and research purposes.
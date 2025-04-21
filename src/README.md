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

## 🧩 Model Architectures

Below is a summary of each model implementation available via the `--model_names` flag.

### 1. Autoregressive (AR)

A simple forecasting baseline that pools information over time before a small feed‑forward network:

```python
def compile_autoregressive(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        GlobalAveragePooling1D(),            # collapses time dimension via mean
        Dense(64, activation='relu'),
        Dense(1, activation='linear')       # single-step forecast
    ])
    return model
```

---

### 2. LSTM

Standard stacked LSTM network:
- Three LSTM layers (32 → 16 → 8 units)
- Dense bottleneck (8 units)
- Linear output

```python
def compile_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True, activation='tanh'),
        LSTM(16, return_sequences=True, activation='tanh'),
        LSTM(8, return_sequences=False, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(1, activation='linear')
    ])
    return model
```

---

### 3. Bidirectional LSTM (BiLSTM)

Symmetric two‑directional LSTM stacking:

```python
def compile_bi_lstm(input_shape):
    model = Sequential([
        Input(input_shape=input_shape),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(16, return_sequences=True)),
        Bidirectional(LSTM(8, return_sequences=False)),
        Dense(8, activation='tanh'),
        Dense(1, activation='linear')
    ])
    return model
```

---

### 4. Simple RNN

Single-layer RNN baseline:

```python
def compile_rnn(input_shape):
    model = Sequential([
        Input(input_shape=input_shape),
        SimpleRNN(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model
```

---

### 5. CNN for Time Series

1D convolution followed by a dense head:

```python
def compile_cnn(input_shape):
    model = Sequential([
        Input(input_shape=input_shape),
        Conv1D(32, kernel_size=3, activation='tanh'),
        Flatten(),
        Dense(8, activation='tanh'),
        Dense(1, activation='linear')
    ])
    return model
```

---

### 6. Attention-LSTM

Integrates self‑attention into an LSTM stack to let the model focus on key time steps:

```python
def compile_attention_lstm(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(32, return_sequences=True),
        SeqSelfAttention(attention_activation='softmax'),
        LSTM(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model
```

**How it works:**
- `SeqSelfAttention` computes a weighted combination of the LSTM outputs at each time step.
- Attention scores (via softmax) highlight important lags before the second LSTM.

---

### 7. Attention-Only

A pure Transformer‑style encoder block for time series:

```python
def compile_attention_only(input_shape):
    inputs = Input(input_shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)
```

**Key points:**
- Multi-head attention attends to multiple temporal patterns in parallel.
- Layer normalization stabilizes training.
- Global pooling collapses the time dimension into a summary vector.

---

### 8. Transformer

Implements a single Transformer encoder block:

```python
def compile_transformer(input_shape):
    inputs = Input(input_shape=input_shape)
    # 1) Self-attention + residual
    attn_out = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = Add()([inputs, attn_out])
    x = LayerNormalization()(x)
    # 2) Feed-forward + residual
    ffn = Dense(64, activation='relu')(x)
    ffn = Dense(input_shape[-1])(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization()(x)
    # 3) Pool & regress
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)
```

**Details:**
- Residual connections (Add layers) help gradient flow.
- Layer normalization after each block stabilizes learning.
- The feed‑forward network expands then projects back to the input dimensionality.

---

### 9. Multi-head LSTM

Combines LSTM with multi‑head attention:

```python
def compile_multihead_lstm(input_shape):
    inputs = Input(input_shape=input_shape)
    x = LSTM(32, return_sequences=True)(inputs)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)
```

---

### 10. Baseline

A trivial predictor that returns the last observed value of the first feature:

```python
class Baseline:
    def fit(self, X, y, *args, **kwargs):
        return self
    def predict(self, X):
        return X[:, -1, 0]
```

---

### 11. Moving Average Baseline

Computes the simple average of the last N steps of the first feature:

```python
class MovingAverageBaseline:
    def __init__(self, window_size=3):
        self.window_size = window_size
    def fit(self, X, y, *args, **kwargs):
        return self
    def predict(self, X):
        N = self.window_size
        if N > X.shape[1]:
            raise ValueError(f"Window size {N} too large for sequence length {X.shape[1]}")
        return X[:, -N:, 0].mean(axis=1)
```

**Usage:**
- Specify via `--model_names MovingAverage`.
- `window_size` defaults to 3 (modify in the `compile_moving_average` wrapper if needed).

# Clean up 

```
make clean 
```



























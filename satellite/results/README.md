# Satellite Results Documentation

This folder contains the analysis results and model outputs from the satellite soil moisture prediction project. Below is a detailed description of each file and its contents.

## Model Performance Files

### `Sat_SM_AMSR_station1_24_6_model_results.csv`
Contains performance metrics for different models predicting AMSR satellite soil moisture data with:
- Input sequence length: 24 hours
- Output prediction length: 6 hours
- Metrics include MAE, MSE, and MAPE for each model type
- Models evaluated: Baseline, Multi-step Linear, Multi-step Dense, CNN, RNN, LSTM, Autoregressive, Bi-LSTM, LSTM_Attention, and BiLSTM_Attention

### `Sat_SM_SMAP_48_12_loss_history.csv`
Training history for SMAP satellite predictions with:
- Input sequence length: 48 hours
- Output prediction length: 12 hours
- Records epoch-by-epoch training and validation losses
- Includes results for Baseline and BiLSTM_Attention models

### `Sat_SM_AMSR_station1_24_6_loss_history.csv`
Detailed training history for AMSR predictions showing:
- Loss values for each epoch during training and validation
- Results for multiple model architectures
- Input length: 24 hours
- Output length: 6 hours

## Analysis Results

### `feature_importance_output.txt`
Contains feature importance analysis results showing:
- Base performance metrics for RNN, CNN, and Custom AR models
- Metrics include MSE, MAE, MAE%, and R² scores
- Separate evaluations for SMAP and AMSR predictions
- Performance impact when excluding different features

### `all_stations_satellite_sm_results.csv`
Comprehensive results across all stations including:
- Model performance metrics by station
- Impact of different meteorological features
- Separate results for SMAP and AMSR predictions
- Metrics broken down by MSE, MAE, MAE percentage, and R²

## Data Format Details

### Performance Metrics
- MSE (Mean Squared Error): Measures average squared difference between predictions and actual values
- MAE (Mean Absolute Error): Average absolute difference between predictions and actual values
- MAPE (Mean Absolute Percentage Error): Percentage representation of prediction errors
- R² (R-squared): Coefficient of determination indicating prediction accuracy

### Model Types
1. Baseline: Simple baseline prediction model
2. Multi-step Linear: Linear regression for multi-step predictions
3. Multi-step Dense: Dense neural network
4. CNN: Convolutional Neural Network
5. RNN: Recurrent Neural Network
6. LSTM: Long Short-Term Memory network
7. Autoregressive: Time series autoregressive model
8. Bi-LSTM: Bidirectional LSTM
9. LSTM_Attention: LSTM with attention mechanism
10. BiLSTM_Attention: Bidirectional LSTM with attention

## Usage Notes

- Loss history files can be used to analyze model convergence and training stability
- Feature importance results help identify key meteorological factors affecting predictions
- Model comparison results assist in selecting the best architecture for specific prediction tasks
- Results are separated by satellite type (AMSR vs SMAP) for comparative analysis

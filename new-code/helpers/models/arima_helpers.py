import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from constants import ARIMA_ORDER

def check_stationarity(df, target_col):
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check if the time series is stationary.
    Returns True if stationary, False otherwise.
    """
    result = adfuller(df[target_col].dropna())
    p_value = result[1]
    print(f"ADF Test for {target_col}: p-value = {p_value:.4f}")
    print("Null hypothesis: Data is non-stationary")
    print(" If p-value < 0.05, reject the null hypothesis and assume the data is stationary.")

    return p_value < 0.05  # If p-value < 0.05, the data is stationary

def train_arima(df, target_col, use_seasonality=True):
    """
    Trains an ARIMA or SARIMA model on the given DataFrame.
    Returns the trained model and predictions.
    """
    df = df.copy()

    # Ensure the index is a DateTimeIndex and set frequency if missing
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.freq is None:
        df = df.asfreq("D")  # Set to "D" for daily data (important for yearly seasonality)

    # Check if data is stationary; apply differencing if needed
    if not check_stationarity(df, target_col):
        print(f"Data for {target_col} is NOT stationary. Applying differencing (d=1).")
        df[target_col] = df[target_col].diff().dropna()  # First-order differencing

    # Split train and test data
    train_size = int(len(df) * 0.8)
    train, test = df[target_col][:train_size], df[target_col][train_size:]

    # Train ARIMA or SARIMA model
    if use_seasonality:
        print("Using Seasonal ARIMA (SARIMA) Model with Yearly Seasonality")
        model = SARIMAX(train,\
                        order=ARIMA_ORDER, \
                        seasonal_order=(1, 1, 1, 365),\
                        maxiter=200)  # 365 days for yearly seasonality
    else:
        print("Using Standard ARIMA Model")
        model = ARIMA(train, order=ARIMA_ORDER)

    model_fit = model.fit()

    # Forecast
    predictions = model_fit.forecast(steps=len(test))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test, label="Actual")
    plt.plot(test.index, predictions, label="Predicted", linestyle="dashed")
    plt.legend()
    plt.title(f"ARIMA Predictions for {target_col} (Yearly Seasonality)")
    plt.show()

    print(f"ARIMA model trained successfully for {target_col}.")
    return model_fit, predictions

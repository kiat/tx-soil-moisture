# System 
import os
import datetime
import IPython
import IPython.display
import argparse

# Suppress TensorFlow INFO/WARN (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# General
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Scikit-learn / SciPy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import ParameterGrid

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional

# Time-series (classical)
# Core statsmodels / pmdarima used in the pipeline
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Optional: alternative forecasting helpers (only if installed / used)
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA as SF_AutoARIMA
    from utilsforecast.plotting import plot_series
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae, mse, rmse, mape
except Exception:
    # Safe to ignore if not using these in pipeline/workflow
    pass

# Gradient Boosting
from xgboost import XGBRegressor

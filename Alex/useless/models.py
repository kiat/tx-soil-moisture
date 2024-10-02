import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
# import shap


#  proprocess the data
def load_and_preprocess_data(file_path):
    # load data
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    data = data.drop(columns=['Latitude', 'Longitude'])
    data = data.drop(columns=['SWC_10', 'SWC_20', 'SWC_50'])
    data = data.fillna(method='ffill')  # forward-fill missing values
    
    features = data.drop(columns=['SWC_5'])  # input features without swc_5
    target = data['SWC_5']  # target is swc_5
    
    # normalize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target.to_numpy(), scaler


# split the data into training and test sets
def split_data(features_scaled, target, input_window=24, output_window=1):
    X, y = [], []
    for i in range(len(features_scaled) - input_window - output_window):
        X.append(features_scaled[i:i+input_window])  # use scaled features
        y.append(target[i+input_window:i+input_window+output_window])  # swc_5 as target
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


import tensorflow as tf
from helpers.feedback_model import FeedBack, BiFeedBack

def baseline_model(input_shape, output_window):
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, 1]),  # Use last timestep as prediction
        tf.keras.layers.Dense(output_window)  # Predict the entire output_window (24 values)
    ])

def linear_model(input_shape, output_window):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(output_window, input_shape=input_shape))  # Linear model with multiple outputs
    model.compile(optimizer='adam', loss='mse')
    return model

def dense_model(input_shape, output_window):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(output_window))  # Output 24 timesteps
    model.compile(optimizer='adam', loss='mse')
    return model

def cnn_model(input_shape, output_window):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(output_window))  # Predict multiple timesteps (24)
    model.compile(optimizer='adam', loss='mse')
    return model

def rnn_model(input_shape, output_window):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dense(output_window))  # Predict 24 timesteps
    model.compile(optimizer='adam', loss='mse')
    return model

def autoregressive_model(input_shape, num_steps):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(num_steps))  # Predict `num_steps` timesteps (24 in this case)
    model.compile(optimizer='adam', loss='mse')
    return model



# arima model
def arima_model(y_train, y_test, order=(5, 1, 0)):
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    
    predictions = model_fit.forecast(steps=len(y_test))
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {"model_name": "ARIMA Model", "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

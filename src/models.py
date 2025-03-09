import argparse
import os
import csv
import pandas as pd
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, SimpleRNN, Conv1D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


def compile_lstm(input_shape):
    model = Sequential([
        LSTM(16, input_shape=input_shape, return_sequences=True),
        # LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
        # LSTM(64, activation='relu', return_sequences=True),
        # Bidirectional(LSTM(32, return_sequences=False)), 
        # Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model

def compile_bilstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(32, activation='tanh', input_shape=input_shape, return_sequences=True)),
        Bidirectional(LSTM(16, return_sequences=True)),
        Bidirectional(LSTM(8, return_sequences=False)),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model


def compile_rnn(input_shape):
    model = Sequential([
        InputLayer(shape=input_shape),
        SimpleRNN(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model

def compile_cnn(input_shape):
    model = Sequential([
        InputLayer(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='tanh'),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    return model

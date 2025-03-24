import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Bidirectional, SimpleRNN, Conv1D, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from keras_self_attention import SeqSelfAttention

# Standard LSTM model
def compile_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32 , activation='tanh', return_sequences=True),
        LSTM(16, activation='tanh', return_sequences=True),
        LSTM(8, activation='tanh', return_sequences=False),
        Dense(8, activation='tanh'),
        Dense(1, activation='linear')
    ])
    return model

# Bidirectional LSTM model
def compile_bilstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(32, activation='tanh', return_sequences=True)),
        Bidirectional(LSTM(16, return_sequences=True)),
        Bidirectional(LSTM(8, return_sequences=False)),
        Dense(8, activation='tanh'),
        Dense(1, activation='linear')
    ])
    return model

# Simple RNN model
def compile_rnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model

# CNN model for time series
def compile_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='tanh'),
        Flatten(),
        Dense(8, activation='tanh'),
        Dense(1, activation='linear')
    ])
    return model



##########################################################
##########################################################
##########################################################


# Attention-LSTM model
def compile_attention_lstm(input_shape, learning_rate=0.0001):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(32, return_sequences=True),
        SeqSelfAttention(attention_activation='softmax'),
        LSTM(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate),
        metrics=[RootMeanSquaredError()]
    )
    return model

# # LSTM Autoencoder
# def compile_lstm_autoencoder(input_shape, learning_rate=0.001):
#     model = Sequential([
#         LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
#         LSTM(64, activation='relu', return_sequences=False),
#         RepeatVector(input_shape[0]),
#         LSTM(64, activation='relu', return_sequences=True),
#         LSTM(128, activation='relu', return_sequences=True),
#         TimeDistributed(Dense(input_shape[1]))
#     ])
#     model.compile(optimizer=Adam(learning_rate), loss='mse')
#     return model

# # Hybrid Attention-LSTM Autoencoder
# def compile_attention_autoencoder(input_shape, learning_rate=0.001):
#     model = Sequential([
#         InputLayer(input_shape=input_shape),
#         LSTM(64, return_sequences=True),
#         SeqSelfAttention(attention_activation='softmax'),
#         LSTM(32, return_sequences=False),
#         RepeatVector(input_shape[0]),
#         LSTM(32, return_sequences=True),
#         LSTM(64, return_sequences=True),
#         TimeDistributed(Dense(input_shape[1]))
#     ])
#     model.compile(optimizer=Adam(learning_rate), loss='mse')
#     return model

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, RepeatVector, TimeDistributed, Input,
    Bidirectional, SimpleRNN, Conv1D, Flatten, InputLayer,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Dropout, Add
)
from keras_self_attention import SeqSelfAttention


#AR
def compile_autoregressive(input_shape):
    model = Sequential([
        InputLayer(shape=input_shape),
        tf.keras.layers.GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model

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
def compile_bi_lstm(input_shape):
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
def compile_attention_lstm(input_shape):
    model = Sequential([
        InputLayer(shape=input_shape),
        LSTM(32, return_sequences=True),
        SeqSelfAttention(attention_activation='softmax'),
        LSTM(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    return model

def compile_attention_only(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1)(x)
    return Model(inputs, x)

def compile_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = Add()([inputs, x])
    x = LayerNormalization()(x)
    ffn = Dense(64, activation='relu')(x)
    ffn = Dense(input_shape[-1])(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1)(x)
    return Model(inputs, x)



def compile_multihead_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(32, return_sequences=True)(inputs)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)


# NOTE: The following models are special cases handled separately in main.py
def compile_baseline(input_shape):
    return Baseline()

def compile_moving_average(input_shape, window_size=3):
    return MovingAverageBaseline(window_size=window_size)





class Baseline:
    def fit(self, X, y, *args, **kwargs):
        return self  # no training

    def predict(self, X):
        return X[:, -1, 0]

class MovingAverageBaseline:
    def __init__(self, window_size=3):
        self.window_size = window_size

    def fit(self, X, y, *args, **kwargs):
        return self  # Still no training

    def predict(self, X):
        # X shape = (batch_size, time_steps, features)
        N = self.window_size
        if N > X.shape[1]:
            raise ValueError(f"Window size {N} is larger than the input sequence length {X.shape[1]}")
        
        # Take the last N time steps and average across the time axis
        return X[:, -N:, 0].mean(axis=1)




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

def build_original_simple_lstm_model(window_size):
    """Builds a simple LSTM model."""
    model = Sequential([
        InputLayer(input_shape=(window_size, 1)),
        LSTM(32),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# TODO: Need to see effect of increasing dense network in accordance to the number of features
def build_simple_lstm_model(window_size, feature_size=1):
    """Builds a simple LSTM model with multiple features, making the dense layer match the number of ."""
    model = Sequential([
        InputLayer(input_shape=(window_size, feature_size)),
        LSTM(32),
        Dense(feature_size)
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def build_optimized_lstm_model(window_size):
    """Builds an optimized LSTM model with bidirectional layers and dropout."""
    model = Sequential([
        InputLayer(input_shape=(window_size, 1)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        # return inputs + delta
        
        # Return the residual (input + delta)
        return inputs + delta

# Initialization of a Residual LSTM model (training on residuals than the output value)
def build_residual_lstm_model(window_size, feature_size):
    """Builds a residual LSTM model that incorprates residual connections by learning the "residual" rather than the direct mapping """
    model = ResidualWrapper(
        model = Sequential([
            InputLayer(input_shape=(window_size, feature_size)),
            LSTM(32, return_sequences=False),
            GlobalAveragePooling1D(),
            Dense(feature_size, kernel_initializer=tf.zeros_initializer)
        ])
    )
    
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# TODO: Compare Residual lstm model w/ the simple lstm model with multiple features method
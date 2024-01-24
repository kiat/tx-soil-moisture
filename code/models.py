import tensorflow as tf
from helpers.feedback_model import FeedBack

def model_dict():
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=False),  #bug was potentially this return_sequences param set as True
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='tanh')
    ])
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(7,),
                            activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    feedback_model = FeedBack(units=64, out_steps=1, num_features=1)

    model_dict = {
        "rnn": feedback_model,
        "lstm": lstm_model,
        "cnn": cnn_model,
    }
    return model_dict
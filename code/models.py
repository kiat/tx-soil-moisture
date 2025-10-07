import tensorflow as tf
from helpers.feedback_model import FeedBack, BiFeedBack


def model_dict(shape):
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, activation = 'relu', input_shape=(shape[0], 21), return_sequences=True),
        tf.keras.layers.LSTM(64, activation = 'relu', return_sequences=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)), 
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='tanh')
    ])
    # bi_lstm_model = tf.keras.models.Sequential([
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation = 'relu', input_shape=(shape[0], 21), return_sequences=True)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dense(units=1, activation='tanh'),
    # ])
    # cnn_model = tf.keras.Sequential([
    #     tf.keras.layers.Conv1D(filters=32,
    #                         kernel_size=(7,),
    #                         activation='relu'),
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=1),
    # ])

    # feedback_model = FeedBack(units=128, out_steps=1, num_features=1)
    # bi_feedback_model = BiFeedBack(units=128, out_steps=1, num_features=1)

    model_dict = {
        # "ar_bi_lstm": bi_feedback_model,
        # "ar_lstm": feedback_model,
        "lstm": lstm_model,
        # "bi_lstm": bi_lstm_model,
        # "cnn": cnn_model,
    }
    return model_dict





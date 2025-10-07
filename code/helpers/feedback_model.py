import tensorflow as tf


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []

        # Initialize the LSTM state.
        x, *state = self.lstm_rnn(inputs)

        # This is the prediction for the first time step
        prediction = self.dense(x)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                              training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
    

class BiFeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        # Use a Bidirectional LSTM layer
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, return_state=True))
        self.lstm_cell = tf.keras.layers.LSTMCell(units * 2)  # *2 because of concatenation of forward and backward states
        self.dense = tf.keras.layers.Dense(num_features)

    def call(self, inputs, training=None):
        predictions = []

        # Process the input sequence with the bidirectional LSTM.
        x, forward_h, forward_c, backward_h, backward_c = self.bidirectional_lstm(inputs)

        # Concatenate the forward and backward states
        state = [tf.keras.layers.concatenate([forward_h, backward_h]),
                 tf.keras.layers.concatenate([forward_c, backward_c])]

        # Initialize the first input of the generative process with the last output of the bidirectional LSTM
        prediction = self.dense(x[:, -1, :])

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            x, state = self.lstm_cell(prediction, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
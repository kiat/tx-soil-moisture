import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load the data from a CSV file
def load_data(filepath):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)  # Parse first column as datetime and use it as index
    return data

# Preprocess the data - Normalize and reshape for the model (now with 24 hours as features)
def preprocess_data(data, feature_columns):
    # Select the 24 features (assuming you have columns that represent each hour of the day)
    train_data = data[feature_columns].values  # Select the 24 columns that represent 24-hour features
    train_data = (train_data - train_data.mean(axis=0)) / train_data.std(axis=0)  # Normalize each feature
    return train_data

# Function to create input/output sequences (input_steps = number of days, output_steps = predict the next day)
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:i + input_steps])  # n days as input
        y.append(data[i + input_steps:i + input_steps + output_steps])  # predict the next day
    return np.array(X), np.array(y)

# Define models
def create_baseline_model(input_steps, feature_size, output_steps):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_steps, feature_size)),
        layers.Lambda(lambda x: x[:, -output_steps:, :])  # Output the last 'output_steps' time steps (next day)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_rnn_model(input_steps, feature_size, output_steps):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_steps, feature_size)),
        layers.SimpleRNN(32, return_sequences=False),  # Output only the final result
        layers.Dense(output_steps * feature_size)  # Predict next day (24 hours)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_cnn_model(input_steps, feature_size, output_steps):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_steps, feature_size)),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(output_steps * feature_size)  # Predict next day (24 hours)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_autoregressive_model(input_steps, feature_size, output_steps):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_steps, feature_size)),
        layers.LSTM(32, return_sequences=False),  # Output only the final result
        layers.Dense(output_steps * feature_size)  # Predict next day (24 hours)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return history

# Main function
def main():
    filepath = '../datasets/Revised_Final_Data/Station1_Revised_Final_Data.csv'  # Correct path to your dataset
    data = load_data(filepath)

    # Define feature columns for 24 hours (these should be your 24-hour feature columns)
    feature_columns = ['T_5', 'T_10', 'T_20', 'T_50', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'Srad', 
                       'SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']  # Example, modify according to your dataset

    # Preprocess the data
    train_data = preprocess_data(data, feature_columns)

    # Create sequences
    input_steps = 7  # Train the model on 7 days
    output_steps = 1  # Predict the next day (1 day, which includes 24 hours)
    X_train, y_train = create_sequences(train_data, input_steps, output_steps)

    # Reshape y_train to match the predicted shape (flattened, since we're predicting 24 hours for multiple features)
    y_train = y_train.reshape(y_train.shape[0], -1)

    # Train different models on the same input-output setup
    models = {
        'Baseline': create_baseline_model(input_steps, len(feature_columns), output_steps),
        'RNN': create_rnn_model(input_steps, len(feature_columns), output_steps),
        'CNN': create_cnn_model(input_steps, len(feature_columns), output_steps),
        'Autoregressive': create_autoregressive_model(input_steps, len(feature_columns), output_steps)
    }

    for model_name, model in models.items():
        print(f"Training {model_name} Model")
        train_model(model, X_train, y_train)

if __name__ == "__main__":
    main()

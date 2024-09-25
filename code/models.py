import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA



def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Drop latitude and longitude as they are static features
    data = data.drop(columns=['Latitude', 'Longitude'])
    
    # Fill any missing values if necessary
    data = data.fillna(method='ffill')
    
    # Normalization (optional based on your need)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler


# Function to split the data into training and test sets
def split_data(data, input_window=24, output_window=1):
    X, y = [], []
    for i in range(len(data) - input_window - output_window):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_window, 1])  # Assuming SWC_5 is the target
    
    return np.array(X), np.array(y)

def split_data_multistep(data, input_window=24, output_steps=3):
    X, y = [], []
    for i in range(len(data) - input_window - output_steps):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_steps, 1])  # Predicting SWC_5 for multiple steps
    
    return np.array(X), np.array(y)


# Define baseline model
def baseline_model(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, 1])  # Takes the last SWC_5 value as the prediction
    ])


# Define linear model
def linear_model(input_shape):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape))  # Linear model with a single neuron
    model.compile(optimizer='adam', loss='mse')
    return model


# Define dense model
def dense_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# Define multistep dense model (for multiple time steps prediction)
def multistep_dense_model(input_shape, output_steps=3):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dense(output_steps))  # Outputs predictions for multiple time steps
    model.compile(optimizer='adam', loss='mse')
    return model

# Define CNN model
def cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# Define RNN model
def rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# Define autoregressive model
def autoregressive_model(input_shape, num_steps):
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# arima model
def arima_model(y_train, y_test, order=(5, 1, 0)):
    """
    Fits ARIMA model to the training data and evaluates it on the test data.
    Args:
        y_train: Training data (target variable)
        y_test: Test data (target variable)
        order: (p, d, q) parameters for ARIMA
    Returns:
        A dictionary with evaluation metrics.
    """
    print(f"Training and evaluating ARIMA model with order: {order}")
    
    # Fit ARIMA model
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(y_test))
    
    # Evaluate the model on test data
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Print individual model results
    print(f"Test MSE: {mse}, Test RMSE: {rmse}, Test MAE: {mae}, R-squared: {r2}\n")
    
    # Return all performance metrics
    return {"model_name": "ARIMA Model", "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# Train and evaluate model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, epochs=50):
    # Print the current model being trained
    print(f"Training and evaluating model: {model_name}")

    # Compile the model if it's not compiled yet
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model on the test data
    test_loss, test_mae = model.evaluate(X_test, y_test)
    
    # Print individual model results
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}\n")
    
    # Return the performance metrics
    return {"model_name": model_name, "loss": test_loss, "mae": test_mae}


# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    data, scaler = load_and_preprocess_data('../datasets/Revised_Final_Data/Station1_Revised_Final_Data.csv')  # replace with actual csv path
    
    # Split data into train and test sets
    X, y = split_data(data)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Get the input shape for the models
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_performance = []

    # Baseline model
    baseline = baseline_model(input_shape)
    result = train_and_evaluate_model(baseline, X_train, y_train, X_test, y_test, model_name="Baseline Model")
    model_performance.append(result)

    # Linear model
    linear = linear_model(input_shape)
    result = train_and_evaluate_model(linear, X_train, y_train, X_test, y_test, model_name="Linear Model")
    model_performance.append(result)

    # Dense model
    dense = dense_model(input_shape)
    result = train_and_evaluate_model(dense, X_train, y_train, X_test, y_test, model_name="Dense Model")
    model_performance.append(result)

    # # Multistep dense model
    # multistep_dense = multistep_dense_model(input_shape, output_steps=3)
    # result = train_and_evaluate_model(multistep_dense, X_train, y_train, X_test, y_test, model_name="Multistep Dense Model")
    # model_performance.append(result)

    # CNN model
    cnn = cnn_model(input_shape)
    result = train_and_evaluate_model(cnn, X_train, y_train, X_test, y_test, model_name="CNN Model")
    model_performance.append(result)

    # RNN model
    rnn = rnn_model(input_shape)
    result = train_and_evaluate_model(rnn, X_train, y_train, X_test, y_test, model_name="RNN Model")
    model_performance.append(result)

    # Autoregressive model
    autoregressive = autoregressive_model(input_shape, num_steps=3)  # Assuming autoregressive for 3 steps
    result = train_and_evaluate_model(autoregressive, X_train, y_train, X_test, y_test, model_name="Autoregressive Model")
    model_performance.append(result)


    # ARIMA
    y_train_arima = y_train[:, 0]  # Assuming you're using the first time step of SWC_5 for ARIMA
    y_test_arima = y_test[:, 0]

    result = arima_model(y_train_arima, y_test_arima, order=(5, 1, 0))  # Using ARIMA with (p=5, d=1, q=0)
    model_performance.append(result)

    print("\nModel Performance Summary:")
    for performance in model_performance:
        if 'loss' in performance:  # Neural network models have 'loss'
            print(f"Model: {performance['model_name']} - Loss: {performance['loss']:.4f}, MAE: {performance['mae']:.4f}")
        else:  # ARIMA doesn't have 'loss'
            print(f"Model: {performance['model_name']} - MSE: {performance['mse']:.4f}, RMSE: {performance['rmse']:.4f}, "
                f"MAE: {performance['mae']:.4f}, R²: {performance['r2']:.4f}")
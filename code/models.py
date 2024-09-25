import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import shap


def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Drop latitude and longitude as they are static features
    data = data.drop(columns=['Latitude', 'Longitude'])
    
    # Drop SWC_10, SWC_20, and SWC_50 (only keep SWC_5 as label)
    data = data.drop(columns=['SWC_10', 'SWC_20', 'SWC_50'])
    
    # Fill any missing values if necessary
    data = data.fillna(method='ffill')
    
    # Separate the SWC_5 column (label) from the features
    features = data.drop(columns=['SWC_5'])  # Input features without SWC_5
    target = data['SWC_5']  # Target is SWC_5
    
    # Normalize features (optional based on your need)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)  # Scale features only
    
    # Return scaled features and target as separate arrays
    return features_scaled, target.to_numpy(), scaler


# Function to split the data into training and test sets
def split_data(features_scaled, target, input_window=24, output_window=1):
    X, y = [], []
    num_features = features_scaled.shape[1]  # Number of features in the dataset (excluding SWC_5)
    
    for i in range(len(features_scaled) - input_window - output_window):
        # Select input features (already scaled)
        X.append(features_scaled[i:i+input_window])  # Use scaled features
        y.append(target[i+input_window:i+input_window+output_window])  # SWC_5 as target
    
    # Convert X and y into numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Return data with correct 3D shape: (batch_size, input_window, num_features)
    return X, y


# Function to split the data for multi-step prediction
def split_data_multistep(data, input_window=24, output_steps=3):
    X, y = [], []
    num_features = data.shape[1]  # Number of features in the dataset (e.g., 14)
    for i in range(len(data) - input_window - output_steps):
        X.append(data[i:i+input_window])  # Shape: (input_window, num_features)
        y.append(data[i+input_window:i+input_window+output_steps, 1])  # Predicting SWC_5 for multiple steps
    
    # Convert X and y into numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Return data with correct 3D shape: (batch_size, input_window, num_features)
    return X, y


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
    model.add(Flatten()) # might delete later, this is primarily for the SHAP values
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

# shap values
def calculate_shap_values_flatten(model, X_train, X_test, feature_names):
   
    explainer = shap.DeepExplainer(model, X_train)  # Use the original 3D input
    
    shap_values = explainer.shap_values(X_test)

    mean_shap_values = np.mean(np.abs(shap_values[0]), axis=0)

    feature_importance = sorted(zip(feature_names, mean_shap_values.mean(axis=0)), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance based on SHAP values:")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance:.4f}")
    
    return shap_values


# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    features_scaled, target, scaler = load_and_preprocess_data('../datasets/Revised_Final_Data/Station1_Revised_Final_Data.csv') 
    
    X, y = split_data(features_scaled, target)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    model_performance = []

    feature_names = ['Ppt', 'T_5', 'T_10', 'T_20', 'T_50', 'Tair', 'RH', 'Windspeed', 'Winddirection', 'Srad']

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
    # X_test_sample = X_test[:100]
    # shap_values = calculate_shap_values_flatten(dense, X_train, X_test_sample, feature_names)

    # # Multistep dense model
    # multistep_dense = multistep_dense_model(input_shape, output_steps=3)
    # result = train_and_evaluate_model(multistep_dense, X_train, y_train, X_test, y_test, model_name="Multistep Dense Model")
    # model_performance.append(result)

    # # CNN model
    # cnn = cnn_model(input_shape)
    # result = train_and_evaluate_model(cnn, X_train, y_train, X_test, y_test, model_name="CNN Model")
    # model_performance.append(result)

    # # RNN model
    # rnn = rnn_model(input_shape)
    # result = train_and_evaluate_model(rnn, X_train, y_train, X_test, y_test, model_name="RNN Model")
    # model_performance.append(result)

    # # Autoregressive model
    # autoregressive = autoregressive_model(input_shape, num_steps=3)  # Assuming autoregressive for 3 steps
    # result = train_and_evaluate_model(autoregressive, X_train, y_train, X_test, y_test, model_name="Autoregressive Model")
    # model_performance.append(result)


    # # ARIMA
    # y_train_arima = y_train[:, 0]  # Assuming you're using the first time step of SWC_5 for ARIMA
    # y_test_arima = y_test[:, 0]

    # result = arima_model(y_train_arima, y_test_arima, order=(5, 1, 0))  # Using ARIMA with (p=5, d=1, q=0)
    # model_performance.append(result)

    print("\nModel Performance Summary:")
    for performance in model_performance:
        if 'loss' in performance:  # Neural network models have 'loss'
            print(f"Model: {performance['model_name']} - Loss: {performance['loss']:.4f}, MAE: {performance['mae']:.4f}")
        else:  
            print(f"Model: {performance['model_name']} - MSE: {performance['mse']:.4f}, RMSE: {performance['rmse']:.4f}, "
                f"MAE: {performance['mae']:.4f}, R²: {performance['r2']:.4f}")
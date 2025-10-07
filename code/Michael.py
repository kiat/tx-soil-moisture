import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit



shape_dict = {
    "s1": [24*7, 24, 1024], 
    "s2": [24*7, 24*7, 1024],
}
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def load_and_preprocess(path, feature_col):
    # load data
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    
    data = data.drop(columns=['Latitude', 'Longitude'])
    #data = data.drop(columns=['SWC_10', 'SWC_20', 'SWC_50'])
    data = data.fillna(method='ffill')  # forward-fill missing values

    # Feature engineering for wind vector and year/days
    wv = data.pop('Windspeed')

    # # Convert to radians.
    wd_rad = data.pop('Winddirection')*np.pi / 180

    # Calculate the wind x and y components.
    data['Wx'] = wv*np.cos(wd_rad)
    data['Wy'] = wv*np.sin(wd_rad)

    timestamp_s = (data.index).map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365.2425)*day

    data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    features = data.drop(columns=[feature_col])  # input features without swc_5
    target = data[feature_col]  # target is swc_5
    
    # normalize features
    #from sklearn.preprocessing import MinMaxScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target.to_numpy(), scaler


# split the data into training and test sets
# modify window parameters to determine input window size and label window size
def split_data(features_scaled, target, input_window, output_window):
    X, y = [], []
    for i in range(len(features_scaled) - input_window - output_window):
        X.append(features_scaled[i:i+input_window])  # use scaled features
        y.append(target[i+input_window:i+input_window+output_window])  # swc_5 as target
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def baseline_model(input_shape, output_shape, label_index=None):
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -output_shape:, 1])  # predicts last swc_5 value
    ])
    

def compile_and_fit(model, train_dataset, model_name, val_dataset=None, patience=2, epochs=10):
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min',
                                                      restore_best_weights=True)

    # Compile the model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    # Train the model using the provided dataset
    history = model.fit(train_dataset, 
                        validation_data=val_dataset, 
                        epochs=epochs, 
                        callbacks=[early_stopping],
                        verbose=1)

    if val_dataset:
        val_loss, val_mae = model.evaluate(val_dataset, verbose=0)
        print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")
    else:
        val_loss, val_mae = None, None

    # Return the history and final evaluation results
    return {
        "model_name": model_name,
        #"history": history,
        "loss": val_loss,
        "mae": val_mae
    }

def linear_model(input_shape, output_steps, num_features):
    model = tf.keras.Sequential([
        # Extract the last time step
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),  # Shape: [batch, 1, features]
        tf.keras.layers.Dense(output_steps)  # Output shape: [batch, output_steps]
    ])
    return model

def dense_model(input_shape, output_steps, num_features):
    # dense = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=64, activation='relu', input_shape = input_shape),
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=output_steps)
    # ])
    # return dense
    dense = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),  # Input shape: (24, 68)
        tf.keras.layers.Flatten(),  # Flatten the input to 1D
        tf.keras.layers.Dense(units=64, activation='relu'),  # First hidden layer
        tf.keras.layers.Dense(units=64, activation='relu'),  # Second hidden layer
        tf.keras.layers.Dense(units=output_steps)  # Final layer outputs `output_steps`
    ])
    return dense

def cnn_model(input_shape, output_steps):
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(3,),
                            activation='relu', input_shape = input_shape),
        tf.keras.layers.Flatten(),  # Flatten the input to 1D
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=output_steps),
    ])
    return conv_model

def lstm_model(input_shape, output_steps):
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape = input_shape),
        # Shape => [batch, time, features]
        tf.keras.layers.Flatten(),  # Flatten the input to 1D
        tf.keras.layers.Dense(units=output_steps)
    ])
    return lstm_model

def bi_lstm_model(input_shape, output_steps):
    bi_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation = 'relu', input_shape=input_shape, return_sequences=True)),
        tf.keras.layers.Flatten(),  # Flatten the input to 1D
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(units=output_steps, activation='tanh'),
    ])
    return bi_lstm_model

# def auto_regressive(input_shape):

#     pass

def create_multi_step_data(X, y, input_steps, output_steps):
    X_multi_step = []
    y_multi_step = []
    
    for i in range(len(X) - input_steps - output_steps + 1):
        # The input will be the past 'input_steps' time steps
        X_multi_step.append(X[i:i + input_steps])
        
        # The output will be the next 'output_steps' time steps
        y_multi_step.append(y[i + input_steps: i + input_steps + output_steps])

    return np.array(X_multi_step), np.array(y_multi_step)

def arima(y_train, y_test, order = (12, 1, 0)):
    arima_model = ARIMA(y_train, order=order)
    arima_model_fit = arima_model.fit()
    
    predictions = arima_model_fit.forecast(steps=len(y_test))
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)


    return {"model_name": "ARIMA Model", "mse": mse, "rmse": rmse, "mae": mae}

def prepare_dataset(X, y, batch_size=32):
    # Create a tf.data.Dataset from features and labels
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    # if shuffle:
    #     dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)

    return dataset

if __name__ == "__main__":

    configurations = [
        {"features": 'SWC_5', "input_steps": 24, "output_steps": 1},
        # {"features": 'SWC_5', "input_steps": 24, "output_steps": 6},
        # {"features": 'SWC_5', "input_steps": 48, "output_steps": 12},
        # {"features": 'SWC_5', "input_steps": 7*24, "output_steps": 24},
        # {"features": 'SWC_5', "input_steps": 7*24, "output_steps": 48},
        # {"features": 'SWC_10', "input_steps": 24, "output_steps": 1},
        # {"features": 'SWC_10', "input_steps": 24, "output_steps": 6},
        # {"features": 'SWC_10', "input_steps": 48, "output_steps": 12},
        # {"features": 'SWC_10', "input_steps": 7*24, "output_steps": 24},
        # {"features": 'SWC_10', "input_steps": 7*24, "output_steps": 48},
        # {"features": 'SWC_20', "input_steps": 24, "output_steps": 1},
        # {"features": 'SWC_20', "input_steps": 24, "output_steps": 6},
        # {"features": 'SWC_20', "input_steps": 48, "output_steps": 12},
        # {"features": 'SWC_20', "input_steps": 7*24, "output_steps": 24},
        # {"features": 'SWC_20', "input_steps": 7*24, "output_steps": 48},
    ]
    # Prepare your data
    # features_scaled, target, scaler = load_and_preprocess('../datasets/Revised_Final_Data/Station3_Revised_Final_Data.csv', features) 

    # output_steps = 4
    # X, y = split_data(features_scaled, target, 24, output_steps)

    # Prepare lists to store performance results across folds
   # model_performance_cv = []
    model_performance_all = []
    for config in configurations:
        print(f"\nRunning configuration: {config}")
        
        features = config['features']
        input_steps = config['input_steps']
        output_steps = config['output_steps']

        # Load and preprocess your data based on selected features
        features_scaled, target, scaler = load_and_preprocess('../datasets/Revised_Final_Data/Station3_Revised_Final_Data.csv', features)

        # Split the data into input and target arrays
        X, y = split_data(features_scaled, target, input_steps, output_steps)

        # Split data into train and test sets (80% train, 20% test)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Prepare datasets for the models
        train_dataset = prepare_dataset(X_train, y_train, batch_size=32)
        test_dataset = prepare_dataset(X_test, y_test, batch_size=32)

        input_shape = (X_train.shape[1], X_train.shape[2])

        # Prepare to store performance results for this configuration
        model_performance = []

        # Baseline model
        # baseline = baseline_model(input_shape, output_steps)
        # baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
        # test_loss, test_mae = baseline.evaluate(test_dataset)
        # model_performance.append({"model_name": "Baseline", "loss": test_loss, "mae": test_mae})

        # # Linear model
        # linear = linear_model(input_shape, output_steps, input_shape[1])
        # result = compile_and_fit(linear, train_dataset, "Linear Model", test_dataset)
        # model_performance.append(result)

        # # Dense model
        # dense = dense_model(input_shape, output_steps, input_shape[1])
        # result = compile_and_fit(dense, train_dataset, "Dense Model", test_dataset)
        # model_performance.append(result)

        # # CNN model
        # cnn = cnn_model(input_shape, output_steps)
        # result = compile_and_fit(cnn, train_dataset, "CNN Model", test_dataset)
        # model_performance.append(result)

        # RNN model
        rnn = lstm_model(input_shape, output_steps)
        result = compile_and_fit(rnn, train_dataset, "RNN Model", test_dataset)
        model_performance.append(result)

        # ARIMA model (if applicable)
        y_train_arima = y_train[:, 0]  # Assuming you're using the first time step of SWC_5 for ARIMA
        y_test_arima = y_test[:, 0]
        result = arima(y_train_arima, y_test_arima, order=(24, 1, 0))  # Using ARIMA with (p=24, d=1, q=4)
        model_performance.append(result)

        # Store the performance of this configuration
        model_performance_all.append({"configuration": config, "performance": model_performance})

    # Output the model performance summary after all configurations
    print("\nModel Performance Summary Across All Configurations:")
    for config_result in model_performance_all:
        print(f"\nConfiguration: {config_result['configuration']}")
        for performance in config_result['performance']:
            if 'loss' in performance:  # Neural network models have 'loss'
                print(f"Model: {performance['model_name']} - Loss: {performance['loss']:.4f}, MAE: {performance['mae']:.4f}")
            else:
                print(f"Model: {performance['model_name']} - MSE: {performance['mse']:.4f}, RMSE: {performance['rmse']:.4f}, "
                    f"MAE: {performance['mae']:.4f}")
    # for config in configurations:
    #     print(f"\nRunning configuration: {config}")
        
    #     features = config['features']
    #     input_steps = config['input_steps']
    #     output_steps = config['output_steps']

    #     # Load and preprocess your data based on selected features
    #     features_scaled, target, scaler = load_and_preprocess('../datasets/Revised_Final_Data/Station3_Revised_Final_Data.csv', features)

    #     # Split the data
    #     X, y = split_data(features_scaled, target, input_steps, output_steps)

    #     # Initialize TimeSeriesSplit
    #     tscv = TimeSeriesSplit(n_splits=n_splits)

    #     # Prepare to store performance results for this configuration
    #     model_performance_cv = []
    #     # Loop over the TimeSeriesSplit folds
    #     for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    #         print(f"Running Fold {fold + 1}...")

    #         # Split the data into training and validation sets for this fold
    #         X_train, X_test = X[train_index], X[test_index]
    #         y_train, y_test = y[train_index], y[test_index]

    #         # Prepare datasets
    #         train_dataset = prepare_dataset(X_train, y_train, batch_size=32)
    #         val_dataset = prepare_dataset(X_test, y_test, batch_size=32) 

    #         input_shape = (X_train.shape[1], X_train.shape[2])
    #         print(X_train.shape)
    #         # Initialize performance list for this fold
    #         fold_performance = []

    #         # Baseline model
    #         baseline = baseline_model(input_shape, output_steps)
    #         baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
    #         test_loss, test_mae = baseline.evaluate(val_dataset)
    #         fold_performance.append({"model_name": "Baseline", "loss": test_loss, "mae": test_mae})

    #         # Linear model
    #         print(input_shape[1])
    #         linear = linear_model(input_shape, output_steps, input_shape[1])
    #         # dummy_input = tf.random.normal([32, 24, 17])  # [batch_size, time, features]

    #         # # Get the model output
    #         # output = linear(dummy_input)

    #         # # Print the output shape
    #         # print(f"Output shape: {output.shape}")  # Should be [32, 5, 17]
    #         result = compile_and_fit(linear, train_dataset, "Linear Model", val_dataset)
    #         fold_performance.append(result)

    #         # Dense model
    #         dense = dense_model(input_shape, output_steps, input_shape[1])
    #         result = compile_and_fit(dense, train_dataset, "Dense Model", val_dataset)
    #         fold_performance.append(result)

    #         # CNN model
    #         cnn = cnn_model(input_shape, output_steps)
    #         result = compile_and_fit(cnn, train_dataset, "CNN Model", val_dataset)
    #         fold_performance.append(result)

    #         # RNN model
    #         rnn = lstm_model(input_shape, output_steps)
    #         result = compile_and_fit(rnn, train_dataset, "RNN Model", val_dataset)
    #         fold_performance.append(result)

    #         # ARIMA model
    #         y_train_arima = y_train[:, 0]  # Assuming you're using the first time step of SWC_5 for ARIMA
    #         y_test_arima = y_test[:, 0]
    #         result = arima(y_train_arima, y_test_arima, order=(24, 1, 4))  # Using ARIMA with (p=24, d=1, q=4)
    #         fold_performance.append(result)

    #         # Store the performance of this fold
    #         model_performance_cv.append({"fold": fold + 1, "performance": fold_performance})

    # # Output the cross-validation performance
    # print("\nCross-Validation Model Performance Summary:")
    # for config_result in model_performance_all:
    #     print(f"\nConfiguration: {config_result['configuration']}")
    #     for fold_performance in config_result['performance']:
    #         print(f"\nFold {fold_performance['fold']} Performance:")
    #         for performance in fold_performance['performance']:
    #             if 'loss' in performance:  # Neural network models have 'loss'
    #                 print(f"Model: {performance['model_name']} - Loss: {performance['loss']:.4f}, MAE: {performance['mae']:.4f}")
    #             else:
    #                 print(f"Model: {performance['model_name']} - MSE: {performance['mse']:.4f}, RMSE: {performance['rmse']:.4f}, "
    #                     f"MAE: {performance['mae']:.4f}, R²: {performance['r2']:.4f}")

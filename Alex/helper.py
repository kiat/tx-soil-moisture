import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_and_preprocess_data(station, target_col, train_split, val_split, window_size, shift_amt, batch_size, data_path="../datasets/Simulate_Cleaned_Merged"):
    # Load data
    station_filepath = f"{data_path}/Station{station}_simulated_cleaned_merged_data.csv"
    dfs = load_data(station_filepath)
    engineer_data(dfs, boolean=False)
    engineer_data(dfs, boolean=True)
    scale_data(dfs)
    cur_df = dfs["cur_station"]

    # Filter columns to keep only the target column and non-SWC columns
    cur_df = cur_df[[col for col in cur_df.columns if not col.startswith('SWC') or col == target_col]]
    # Retain only columns that either start with 'SWC' or match the target column
    cur_df = cur_df[[col for col in cur_df.columns if not col.startswith('SWC') or col == target_col]]

    # Define columns to drop, including 'T_10', 'T_20', 'T_50', 'latitude', and 'longitude'
    columns_to_drop = [col for col in cur_df.columns if col in ["T_10", "T_20", "T_50", "Latitude", "Longitude"]]

    # Drop the specified columns from the DataFrame
    cur_df.drop(columns=columns_to_drop, inplace=True)

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        cur_df, target_col, train_split, val_split, window_size, shift_amt
    )

    # Generate batches for training, validation, and testing
    train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=batch_size)
    val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=batch_size)
    test_dataset, test_steps = generate_batches(X_test, y_test, batch_size=batch_size)

    return (
        cur_df, 
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_dataset, val_dataset, test_dataset, 
        train_steps, val_steps, test_steps
    )

# function to load and preprocess the data
def load_data(filepath):
    dfs = {}
    df = pd.read_csv(filepath, sep=",", parse_dates=["Unnamed: 0"], index_col="Unnamed: 0")
    dfs['cur_station'] = df
    return dfs

def engineer_data(dfs, boolean):
    day = 24 * 60 * 60
    year = 365.2425 * day

    for station, df in dfs.items():
        df = df.dropna()
        wv = df['Windspeed']
        wd_rad = np.deg2rad(df['Winddirection'])

        df.index = pd.to_datetime(df.index)
        timestamp_s = df.index.map(pd.Timestamp.timestamp)
        lat = np.deg2rad(df['Latitude'])
        lon = np.deg2rad(df['Longitude'])

        if boolean:
            df['Wx'] = wv * np.cos(wd_rad)
            df['Wy'] = wv * np.sin(wd_rad)

        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        df['x_cord'] = np.cos(lat) * np.cos(lon)
        df['y_cord'] = np.cos(lat) * np.sin(lon)
        df['z_cord'] = np.sin(lat)

        dfs[station] = df

    return dfs

def scale_data(dfs):
    for station, df in dfs.items():
        cur_df = df.copy()
        d_sin = cur_df.pop("Day sin")
        d_cos = cur_df.pop("Day cos")
        y_sin = cur_df.pop("Year sin")
        y_cos = cur_df.pop("Year cos")
        x = cur_df.pop("x_cord")
        y = cur_df.pop("y_cord")
        z = cur_df.pop("z_cord")
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(data=scaler.fit_transform(cur_df), columns=cur_df.columns, index=cur_df.index)
        scaled_df["Day sin"] = d_sin.values
        scaled_df["Day cos"] = d_cos.values
        scaled_df["Year sin"] = y_sin.values
        scaled_df["Year cos"] = y_cos.values
        scaled_df["x_cord"] = x.values
        scaled_df["y_cord"] = y.values
        scaled_df["z_cord"] = z.values
        dfs[station] = scaled_df

# data preprocessing
def preprocess_data(df, target_col, train_split, val_split, window_size, shift_amt):

    train_set, val_set, test_set, target_idx = split(df, target_col, train_split, val_split)

    X_train, y_train = generate_windows(train_set, window_size, shift_amt, target_idx)
    X_val, y_val = generate_windows(val_set, window_size, shift_amt, target_idx)
    X_test, y_test = generate_windows(test_set, window_size, shift_amt, target_idx)

    return X_train, y_train, X_val, y_val, X_test, y_test

def split(df, target_col, train_split, val_split):

    target_idx = df.columns.get_loc(target_col)
    train_set = df[:int(len(df) * train_split)].values

    val_set = df[int(len(df) * train_split):int(len(df) * (train_split + val_split))].values
    test_set = df[int(len(df) * (train_split + val_split)):].values

    return train_set, val_set, test_set, target_idx

def generate_windows(data, window_size, shift, target_idx):
    labels = data[:, target_idx]
    X, y = [], []

    for i in range(len(data) - window_size - shift):
        window = data[i:i + window_size]
        window_label = labels[i + window_size + shift]

        X.append(window)
        y.append(window_label)
    return np.array(X), np.array(y)

def generate_batches(X, y, batch_size=32):
    tf_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    tf_dataset = tf_dataset.repeat().batch(batch_size=batch_size, drop_remainder=True)
    steps_per_epoch = len(X) // batch_size
    return tf_dataset, steps_per_epoch

def compile_and_fit(model, data, steps_per_epoch, val_data, val_steps, model_name='model/', patience=3, max_epochs=25, batch_size=32):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    ckpt = tf.keras.callbacks.ModelCheckpoint(model_name + ".keras", save_best_only=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()])
    
    history = model.fit(data, epochs=max_epochs, callbacks=[ckpt, early_stopping],
                        validation_data=val_data, validation_steps=val_steps,
                        shuffle=False, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
    
    return history

def plot_single_pred(model, name, dataset, data_steps, y, batch_size=32):
    forecast = model.predict(dataset, batch_size=batch_size, steps=data_steps)
    if len(forecast.shape) == 3:
        forecast = forecast[:, 0, 0]
    elif len(forecast.shape) == 2:
        forecast = forecast[:, 0]
    plt.figure(figsize=(10, 6))
    plt.plot(y)
    plt.plot(forecast)
    plt.legend(("Actual", "Predictions"))





# define models
bi_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)), # try commenting this out
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='tanh')
])

linear_model = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

dense_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(128, return_sequences=True),
    tf.keras.layers.SimpleRNN(64, return_sequences=True),
    tf.keras.layers.SimpleRNN(32, return_sequences=False),
    tf.keras.layers.Dense(units=1)
])

# bidirectional_model = tf.keras.Sequential([tf.keras.layers.Bidrectional(units=1)])
def create_autoregressive_model(input_shape):
    # Create the autoregressive model, where input_shape is passed from main.py
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=input_shape)
    ])
    return model

def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1)
    ])
    return model


def compare_models(model_results):
    for model_name, mse in model_results.items():
        print(f"{model_name} Model MSE: {mse}")


# Function to run ARIMA model and make predictions
def run_arima_model(train_data, test_data, order=(1, 0, 0)):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()

    # Make predictions on the test data
    predictions = model_fit.forecast(steps=len(test_data))
    
    #mse
    mse = mean_squared_error(test_data, predictions)
    print(f"ARIMA Model MSE: {mse}")
    return mse


def feature_importance_analysis(features, cur_df, target_col, model, model_name, train_split, val_split, window_size, shift_amt, batch_size, patience, max_epochs):
    mse_results = {}
    mae_results = {}
    mape_results = {}

    for feature in features:
        print(f"Training {model_name} without feature: {feature}")
        cur_df_modified = cur_df.drop(columns=[feature])
        
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
            cur_df_modified, target_col, train_split, val_split, window_size, shift_amt
        )
        train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=batch_size)
        val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=batch_size)

        history = compile_and_fit(
            model, train_dataset, train_steps, val_dataset, val_steps,
            batch_size=batch_size, model_name=f"{model_name}_without_{feature}",
            patience=patience, max_epochs=max_epochs
        )

        mse = history.history['val_mean_squared_error'][-1]
        mae = history.history['val_mean_absolute_error'][-1]
        mape = history.history['val_mean_absolute_percentage_error'][-1]

        mse_results[feature] = mse
        mae_results[feature] = mae
        mape_results[feature] = mape
    
    return mse_results, mae_results, mape_results

def run_feature_importance_for_model(cur_df, feature_names, target_col, model, model_name, train_split, val_split, window_size, shift_amt, batch_size, patience, max_epochs):
    mse_results, mae_results, mape_results = feature_importance_analysis(
        feature_names, cur_df, target_col, model, model_name, train_split, val_split, window_size, shift_amt, batch_size, patience, max_epochs
    )

    sorted_mse = sorted(mse_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\nMSE results after removing each feature for {model_name}: (higher indicates more important)")
    for feature, mse in sorted_mse:
        print(f"Feature: {feature}, MSE: {mse}")

    sorted_mae = sorted(mae_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\nMAE results after removing each feature for {model_name}:")
    for feature, mae in sorted_mae:
        print(f"Feature: {feature}, MAE: {mae}")

    sorted_mape = sorted(mape_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\nMAPE results after removing each feature for {model_name}:")
    for feature, mape in sorted_mape:
        print(f"Feature: {feature}, MAPE: {mape}")

def feature_addition_analysis(features, cur_df, target_col, model, model_name, train_split, val_split, window_size, shift_amt, batch_size, patience, max_epochs):
    mse_results = {}
    mae_results = {}
    mape_results = {}

    for feature in features:
        print(f"Training {model_name} with only feature: {feature}")
        cur_df_modified = cur_df[[feature, target_col]]
        
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
            cur_df_modified, target_col, train_split, val_split, window_size, shift_amt
        )
        train_dataset, train_steps = generate_batches(X_train, y_train, batch_size=batch_size)
        val_dataset, val_steps = generate_batches(X_val, y_val, batch_size=batch_size)

        history = compile_and_fit(
            model, train_dataset, train_steps, val_dataset, val_steps,
            batch_size=batch_size, model_name=f"{model_name}_with_only_{feature}",
            patience=patience, max_epochs=max_epochs
        )

        mse = history.history['val_mean_squared_error'][-1]
        mae = history.history['val_mean_absolute_error'][-1]
        mape = history.history['val_mean_absolute_percentage_error'][-1]

        mse_results[feature] = mse
        mae_results[feature] = mae
        mape_results[feature] = mape
    
    return mse_results, mae_results, mape_results

def run_feature_addition_for_model(cur_df, feature_names, target_col, model, model_name, train_split, val_split, window_size, shift_amt, batch_size, patience, max_epochs):
    mse_results, mae_results, mape_results = feature_addition_analysis(
        feature_names, cur_df, target_col, model, model_name, train_split, val_split, window_size, shift_amt, batch_size, patience, max_epochs
    )

    sorted_mse = sorted(mse_results.items(), key=lambda x: x[1])
    print(f"\nMSE results after adding each feature for {model_name}: (lower indicates more important)")
    for feature, mse in sorted_mse:
        print(f"Feature: {feature}, MSE: {mse}")

    sorted_mae = sorted(mae_results.items(), key=lambda x: x[1])
    print(f"\nMAE results after adding each feature for {model_name}:")
    for feature, mae in sorted_mae:
        print(f"Feature: {feature}, MAE: {mae}")

    sorted_mape = sorted(mape_results.items(), key=lambda x: x[1])
    print(f"\nMAPE results after adding each feature for {model_name}:")
    for feature, mape in sorted_mape:
        print(f"Feature: {feature}, MAPE: {mape}")
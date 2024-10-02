import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# function to load and preprocess the data
def load_data(filepath):
    dfs = {}
    df = pd.read_csv(filepath, sep=",", parse_dates=["Unnamed: 0"], index_col="Unnamed: 0")
    dfs['Station1'] = df
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
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
    
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
lstm_model = tf.keras.models.Sequential([ 
    tf.keras.layers.LSTM(128, return_sequences=True), # try commenting this out
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='tanh')
])

bi_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)), # try commenting this out
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='tanh')
])

linear_model = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

autoregressive_model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=(None, 1))])

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf



def consolidate_wind_vectors(df):
    wv = df.pop('Windspeed')

    # Convert to radians.
    wd_rad = df.pop('Winddirection')*np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)
    return df

def add_periodic_time_features(df):
    timestamp_s = (df.index).map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    return df

def fft_visualizer(df):
    fft = tf.signal.rfft(df['Tair'])
    f_per_dataset = np.arange(0, len(fft))

    n_samples_h = len(df['Tair'])
    hours_per_year = 24*365.2524
    years_per_dataset = n_samples_h/(hours_per_year)

    f_per_year = f_per_dataset/years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 400000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')


### Feature Scaling
def normalize_features(train_df, test_df):
    scaler = MinMaxScaler()
    # Fit on the entire set of columns (features + target)
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
        index=train_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
        index=test_df.index
    )
    return train_scaled, test_scaled, scaler




################################







### Fourier Seasonal Feature Engineering
def add_fourier_terms(df, num_terms=3):
    """
    Adds Fourier transform components to capture seasonality.
    """
    N = len(df)
    t = np.arange(N)

    for i in range(1, num_terms + 1):
        df[f"fourier_sin_{i}"] = np.sin(2 * np.pi * i * t / N)
        df[f"fourier_cos_{i}"] = np.cos(2 * np.pi * i * t / N)

    return df
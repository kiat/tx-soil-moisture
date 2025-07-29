import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr

# Tensorflow and Keras for CNN/LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, LSTM

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from pmdarima import auto_arima

from itertools import combinations


'''     Feature Filtering - Manual and Automated       '''
def filter_features(df, manual_features, label_col='y', threshold=0.95, auto_drop_high_corr=False):
    print("\n Starting feature filtering...")

    # Identify all T_ and SWC_ columns to drop (unless manually kept)
    all_columns = df.columns.tolist()
    drop_t_cols = [col for col in all_columns if col.startswith('T_') and col not in manual_features]
    drop_swc_cols = [col for col in all_columns if col.startswith('SWC_') and col not in manual_features]
    df = df.drop(columns=drop_t_cols + drop_swc_cols, errors='ignore')

    # Remove label column before correlation
    features_only = df.drop(columns=[label_col], errors='ignore')

    # Separate manually selected features from the rest
    features_to_test = features_only.drop(columns=manual_features, errors='ignore')

    # Step 4: Optionally drop highly correlated features
    to_drop = []
    if auto_drop_high_corr:
        corr_matrix = features_to_test.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    reduced_features = [col for col in features_to_test.columns if col not in to_drop]
    final_features = manual_features + reduced_features

    print(f"Manually kept: {manual_features}")
    print(f"Dropped other T_* columns: {drop_t_cols}")
    print(f"Dropped other SWC_* columns: {drop_swc_cols}")
    if auto_drop_high_corr:
        print(f"Auto-dropped due to high correlation (> {threshold}): {to_drop}")
    else:
        print("Skipped auto-drop of correlated features")

    print(f"Final kept features: {final_features}")
    return df[final_features]


'''     FFT and Zoomed Plotting       '''
def plot_fft_analysis(swc_series, samples_per_day=24, zoom_xlim=(0.01, 0.1), zoom_ylim=(0, 1000)):
    # zoom_xlim and zoom_ylim parameterize zooming to view seasonality
    swc = np.asarray(swc_series).astype(np.float32)
    fft = tf.signal.rfft(swc)

    # Frequency scaling
    n_samples = len(swc)
    f_per_dataset = np.arange(0, len(fft))
    f_per_day = f_per_dataset * (samples_per_day / n_samples)

    # Plot: full frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.step(f_per_day, np.abs(fft), color='purple')
    plt.xlabel('Frequency (cycles per day)')
    plt.ylabel('Amplitude')
    plt.title('FFT of Time Series (Full Spectrum)')
    plt.xlim([0, 10])
    plt.grid()
    plt.show()

    # Plot: zoomed-in on seasonal frequencies
    plt.figure(figsize=(12, 6))
    plt.step(f_per_day, np.abs(fft), color='darkorange')
    plt.xlabel('Frequency (cycles per day)')
    plt.ylabel('Amplitude')
    plt.title('FFT (Zoom into Seasonal Range)')
    plt.xlim(zoom_xlim)
    plt.ylim(zoom_ylim)
    plt.grid()
    plt.show()


'''     Feature Engineering       '''
def engineer_features(df):
    df = df.copy()

    # Drop unused spatial features
    df = df.drop(columns=['Latitude', 'Longitude'], errors='ignore')

    # Wind vector transformation
    if 'Windspeed' in df.columns and 'Winddirection' in df.columns:
        df['Wind_X'] = df['Windspeed'] * np.cos(np.deg2rad(df['Winddirection']))
        df['Wind_Y'] = df['Windspeed'] * np.sin(np.deg2rad(df['Winddirection']))
        df = df.drop(columns=['Windspeed', 'Winddirection'])

    # Time-based cyclical features
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day

    df['DaySin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['DayCos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['YearSin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['YearCos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


'''     Capturing Seasons       '''
def get_seasonal_data(df, year, season_name, seasons_dict):
    start_suffix, end_suffix = seasons_dict[season_name]
    
    if season_name == "winter":
        start = f"{year - 1}-{start_suffix}"
        end = f"{year}-{end_suffix}"
    else:
        start = f"{year}-{start_suffix}"
        end = f"{year}-{end_suffix}"

    return df[start:end]


'''     Data Windowing       '''
def data_to_X_y(data, window_size, offset):
    X, y = [], []
    for i in range(len(data) - window_size - offset):
        X.append(data[i:i+window_size, :])  
        y.append(data[i + window_size + offset, 0])  

    return  np.array(X),  np.array(y)


'''     CNN and LSTM Building       '''

def build_cnn_model(input_shape, params):
    model = Sequential([
        Conv1D(filters=params["filters"],
               kernel_size=params["kernel_size"],
               activation=params["activation"],
               padding='same',
               input_shape=input_shape),
        GlobalAveragePooling1D(),
        Dense(params["dense_units"], activation=params["activation"]),
        Dropout(params["dropout_rate"]),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(loss='mae', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def build_lstm_model(input_shape, params):
    model = Sequential([
        LSTM(params["lstm_units"], activation=params["activation"], return_sequences=False, input_shape=input_shape),
        Dense(params["dense_units"], activation=params["activation"]),
        Dropout(params["dropout_rate"]),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(loss='mae', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def build_arima_model(params):
    def model_fn(y_train):
        return ARIMA(y_train, order=(params['p'], params['d'], params['q'])).fit()
    return model_fn

def build_arimax_model(params, exog_train):
    def model_fn(y_train):
        return ARIMA(y_train, exog=exog_train, order=(params['p'], params['d'], params['q'])).fit()
    return model_fn


def build_autoarima_model(y_train):
    model = auto_arima(y_train, seasonal=False, stepwise=True, suppress_warnings=True)
    return model

def build_sarima_model(y_train, order, seasonal_order):
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
    return model

def build_xgboost_model(X_train, y_train, X_test, params=None):
    model = XGBRegressor(**(params or {}))
    # Reshape for XGBoost (2D input required)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_train = y_train.reshape(-1)

    model.fit(X_train, y_train)

    return model


def generate_exog_feature_sets(static_features, max_comb=3):
    feature_sets = []
    for r in range(1, min(len(static_features), max_comb) + 1):
        feature_sets.extend(combinations(static_features, r))
    return feature_sets


'''     CNN and LSTM Plotting and Metrics       '''

def plot_predictions(X_test, y_test, model, target_scaler, original_data, label_col="SWC_10"):
    # Predict and reshape
    y_preds = model.predict(X_test).reshape(-1, 1)
    y_actual = y_test.reshape(-1, 1)

    # Inverse transform
    if target_scaler:
        y_preds_inv = target_scaler.inverse_transform(y_preds)
        y_actual_inv = target_scaler.inverse_transform(y_actual)
    else:
        y_preds_inv = y_preds
        y_actual_inv = y_actual


    # Match with correct timestamps from original data
    date_range = original_data.index[-len(y_preds_inv):]

    # Build results DataFrame
    results_df = pd.DataFrame({
        "date": date_range,
        "Actual SWC": y_actual_inv.flatten(),
        "Predicted SWC": y_preds_inv.flatten()
    })

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(results_df["date"], results_df["Actual SWC"], label="Actual SWC", color="blue")
    plt.plot(results_df["date"], results_df["Predicted SWC"], label="Predicted SWC", color="red", alpha=0.7)
    plt.title("SWC Forecast vs Actual (All Batches)")
    plt.xlabel("Time")
    plt.ylabel("SWC_10 (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def report_metrics(X_test, y_test, model, scaler, label_col="SWC_10", return_metrics=True):
    # Flatten predictions and labels
    y_pred = model.predict(X_test).flatten()
    y_true = y_test.flatten()

    # Inverse transform both
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
    corr = pearsonr(y_true_inv, y_pred_inv).statistic

    # Print results
    print(f"\n Evaluation Metrics for {label_col}:")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"MAPE:  {mape * 100:.2f}%")
    print(f"CORR:  {corr:.4f}")

    if return_metrics:
        return {"rmse": rmse, "mae": mae, "mape": mape, "corr": corr}
    


import seaborn as sns
from tabulate import tabulate

def summarize_results(results, metric="rmse"):
    if not results:
        print("No results to display.")
        return

    df = pd.DataFrame(results)
    print(df)

    # Best configuration
    best = df.loc[df[metric].idxmin()]
    print("\n=== Best Model Summary ===")
    print(tabulate([best], headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))

    # Full metrics table
    print("\n=== All Results Summary ===")
    print(tabulate(df[["model", "config", "season", "output_horizon", "rmse", "mae", "mape", "corr"]],
                   headers="keys", tablefmt="github", floatfmt=".4f"))

    # Plot: RMSE by model
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="rmse", errorbar=None)
    plt.title("RMSE by Model")
    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: RMSE by season (if available)
    if "season" in df.columns and df["season"].nunique() > 1:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="season", y="rmse", hue="model")
        plt.title("Seasonal RMSE Distribution by Model")
        plt.ylabel("RMSE")
        plt.xlabel("Season")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# model_runner.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from config import *
from helpers import *

# Global results list and utilities
results = []

def log_results(model_name, config, label, metrics):
    results.append({
        "model": model_name,
        "config": config,
        "label": label,
        **metrics
    })

def get_best_config(primary_metric="rmse", minimize=True):
    if not results:
        return None
    sorted_results = sorted(results, key=lambda x: x[primary_metric], reverse=not minimize)
    return sorted_results[0]

def save_results_to_csv(filepath="evaluation_results.csv"):
    pd.DataFrame(results).to_csv(filepath, index=False)

def get_train_test_split(df, train_start=TRAIN_START_YEAR, train_end=TRAIN_END_YEAR, test_year=TEST_YEAR):
    if not train_start:
        train_start = df.index.min().strftime('%Y-%m-%d')
    if not train_end:
        train_end = '2018-12-31'
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    test_df = df[(df.index >= f"{test_year}-01-01") & (df.index <= f"{test_year}-12-31")]
    return train_df, test_df

def run_forecast(train_df, test_df, season, horizon, model_name, model_params, config_id):
    # for model_name in models:
    print(f"\n=== Running Model: {model_name.upper()} ===")
    requires_windowing = model_meta[model_name]["requires_windowing"]
    param_list = PARAM_GRID[model_name] if TUNE else [MODEL_PARAMS[model_name]]

    # for config_id, model_params in enumerate(param_list, 1):
    print(f"\n→  Params: {model_params}")
    label = f"{model_name.upper()} | Horizon={horizon} | Season={season or 'All'}"

    if requires_windowing:
        train_filtered = filter_features(train_df.copy(), MANUAL_KEEP, label_col=TARGET_COL, threshold=THRESHOLD, auto_drop_high_corr=HIGH_CORR_FILTER)
        test_filtered = filter_features(test_df.copy(), MANUAL_KEEP, label_col=TARGET_COL, threshold=THRESHOLD, auto_drop_high_corr=HIGH_CORR_FILTER)
        target_scaler = StandardScaler()
        train_scaled = target_scaler.fit_transform(train_filtered)
        test_scaled = target_scaler.transform(test_filtered)

        X_train, y_train = data_to_X_y(train_scaled, window_size=INPUT_WINDOW, offset=horizon)
        X_test, y_test = data_to_X_y(test_scaled, window_size=INPUT_WINDOW, offset=horizon)

        if model_name == "cnn":
            model = build_cnn_model(input_shape=X_train.shape[1:], params=model_params)
            model.fit(X_train, y_train, epochs=40)
        elif model_name == "lstm":
            model = build_lstm_model(input_shape=X_train.shape[1:], params=model_params)
            model.fit(X_train, y_train, epochs=40)
        elif model_name == "xgboost":
            model = build_xgboost_model(X_train, y_train, X_test, params=model_params)  # already trained
            X_test = X_test.reshape((X_test.shape[0], -1))


        y_pred = model.predict(X_test)

        TARGET_INDEX = train_filtered.columns.get_loc(TARGET_COL)
        target_mean = target_scaler.mean_[TARGET_INDEX]
        target_std = target_scaler.scale_[TARGET_INDEX]

        # Manually inverse-transform only the target dimension
        y_pred = y_pred.reshape(-1, 1) * target_std + target_mean
        y_true = y_test.reshape(-1, 1) * target_std + target_mean
        # if model_name in ["cnn", "lstm"]:
        #     plot_predictions(X_test, y_test, model, target_scaler=target_scaler, original_data=test_filtered, label_col=TARGET_COL)
    else:
        y_train = train_df[TARGET_COL].values
        y_test = test_df[TARGET_COL].values
        dates = test_df.index

        if model_name == "arimax":
            feature_sets = generate_exog_feature_sets(EXOG_FEATURES) if SEARCH_ARIMAX_FEATURES else [tuple(EXOG_FEATURES)]
            for exog_features in feature_sets:
                print(f"→ Trying ARIMAX with features: {exog_features}")
                exog_train = train_df[list(exog_features)]
                exog_test = test_df[list(exog_features)]
                model = build_arimax_model(model_params, exog_train)(y_train)
                y_pred = model.forecast(steps=len(y_test), exog=exog_test)
                break
        elif model_name == "autoarima":
            model = auto_arima(y_train, seasonal=False, stepwise=True, suppress_warnings=True)
            y_pred = model.predict(n_periods=len(y_test))
        elif model_name == "sarima":
            order = model_params.get("order", (1, 1, 1))
            seasonal_order = model_params.get("seasonal_order", (1, 1, 1, 24))
            model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
            y_pred = model.forecast(steps=len(y_test))
        else:
            model = build_arima_model(model_params)(y_train)
            y_pred = model.forecast(steps=len(y_test))
        y_true = y_test
        # if model_name not in ['cnn', 'lstm', 'xgboost']:
        #     plot_predictions(y_true=y_true, y_pred=y_pred, dates=dates, title=f"{label}")

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    corr = pearsonr(y_true.flatten(), y_pred.flatten()).statistic
    features_used = list(exog_features) if model_name == "arimax" else (MANUAL_KEEP if model_name in ["cnn", "lstm", "xgboost"] else None)

    log_results(model_name, config_id, label, {
        "params": model_params,
        "output_horizon": horizon,
        "season": season or "All",
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "corr": corr,
        "features": features_used
    })

def run_pipeline(data, models = MODELS_TO_RUN):
    for model_name in models:
        print(f"\n=== Running Model: {model_name.upper()} ===")
        configs = PARAM_GRID[model_name] if TUNE else [MODEL_PARAMS[model_name]]

        for i, params in enumerate(configs):
            label_base = f"{model_name}-tuned{i+1}" if TUNE else model_name
            print(f"\n→ Config {i+1} | Params: {params}")

            if TEST_SEASONS and TEST_FULL:
                train_df, _ = get_train_test_split(data)
                for season in SEASONS:
                    seasonal_df = get_seasonal_data(data, TEST_YEAR, season, SEASONS)
                    if len(seasonal_df) < INPUT_WINDOW + max(OUTPUT_HORIZONS):
                        print(f"Skipping {season}: not enough data")
                        continue
                    for h in OUTPUT_HORIZONS:
                        run_forecast(train_df, seasonal_df, season=season, horizon=h, model_name=model_name, model_params=params, config_id=i+1)

            elif TEST_SEASONS:
                train_df, _ = get_train_test_split(data)
                for season in SEASONS:
                    seasonal_df = get_seasonal_data(data, TEST_YEAR, season, SEASONS)
                    if len(seasonal_df) < INPUT_WINDOW + OUTPUT_HORIZON:
                        print(f"Skipping {season}: not enough data")
                        continue
                    run_forecast(train_df, seasonal_df, season=season, horizon=h, model_name=model_name, model_params=params, config_id=i+1)

            elif TEST_FULL:
                for h in OUTPUT_HORIZONS:
                    train_df, test_df = get_train_test_split(data)
                    run_forecast(train_df, test_df, season=None, horizon=h, model_name=model_name, model_params=params, config_id=i+1)

            else:
                train_df, test_df = get_train_test_split(data)
                run_forecast(train_df, test_df, season=None, horizon=OUTPUT_HORIZON, model_name=model_name, model_params=params, config_id=i+1)

    if LOG_TO_CSV:
        save_results_to_csv()

    best = get_best_config()
    if best:
        summarize_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Soil Moisture Forecasting Models")
    parser.add_argument('--models', type=str, help="Comma-separated list of models to run")
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning")
    parser.add_argument('--seasons', action='store_true', help="Enable seasonal evaluation")
    parser.add_argument('--full', action='store_true', help="Enable testing over all output horizons")
    parser.add_argument('--log', action='store_true', help="Log results to CSV")
    args = parser.parse_args()

    if args.models:
        models = args.models.split(',')
    if args.tune:
        TUNE = True
    if args.seasons:
        TEST_SEASONS = True
    if args.full:
        TEST_FULL = True
    if args.log:
        LOG_TO_CSV = True

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"], index_col="Date")
    df = engineer_features(df)
    run_pipeline(df)

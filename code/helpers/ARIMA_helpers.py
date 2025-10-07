from helpers.data_helpers import retrieve_data
import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_arima(target_feat = "SWC_5", exo_feats = None, seasonal = False, stepwise = False):
    buff, train_df = retrieve_data()
    test_df = retrieve_data(test = True)
    end_test = test_df[target_feat].values
    end_train = train_df[target_feat].values
    if exo_feats == None:
        exo_train = None
    else: 
        exo_train = train_df[exo_feats].values
    print("Starting Model")
    model = pm.auto_arima(end_train, 
                          exogenous = exo_train,
                          trace = True,
                          start_p=1,
                          start_q=1,  
                          max_p=7, # Change later
                          max_q=7, # Change
                          m=8670, # is the frequncy of the cycle
                          seasonal=seasonal, 
                          d=1,
                          #D=None,
                          #start_P=0, 
                          #start_Q=0, 
                          #max_P = 1, 
                          #max_Q = 1, 
                          #information_criteria = 'AIC',
                          #error_action='ignore',
                          #suppress_warnings=True,
                          stepwise=stepwise)  #Trying False
    print("Model Ran")
    if exo_feats == None:
        forecast = model.predict(len(end_test))
    else:
        forecast = model.predict(n_periods=len(end_test), exogenous=test_df[exo_feats])
    print("Results For ARIMA")
    mse=mean_squared_error(forecast, end_test)
    mae = mean_absolute_error(forecast, end_test)
    rmse = np.sqrt(mean_squared_error(forecast, end_test))
    mape = np.mean(np.abs((forecast - end_test) / forecast)) * 100 
    print('\n')
    print(mse, mae, rmse, mape)
    return forecast


def add_baseline(folder, target_feat = "SWC_5", exo_feats = None, SARIMAX = False):
    arima_forecast = run_arima(target_feat, exo_feats, False)
    results = pd.read_csv(folder + "/results.csv")
    max_len = results.shape[0]
    arima_len = len(arima_forecast)
    results["ARIMA"] = arima_forecast[arima_len - max_len:]
    if SARIMAX:
        sarimax_forecast = run_arima(target_feat, exo_feats, True)
        results["SARIMAX"] = sarimax_forecast[arima_len - max_len:]

    results.to_csv(folder + "/results.csv", index = False)

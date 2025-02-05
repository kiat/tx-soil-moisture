import statsmodels.api as sm

def fit_sarima_model(data, order=(5, 1, 2), seasonal_order=(1, 0, 1, 24)):
    """
    Fits a SARIMA model to the provided 1D data.
    
    Parameters:
        data (array-like or pandas Series): The time series data to fit.
        order (tuple): The (p, d, q) order for the ARIMA part.
        seasonal_order (tuple): The (P, D, Q, s) order for the seasonal part.
    
    Returns:
        results: The fitted SARIMA model results.
    """
    model = sm.tsa.statespace.SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results
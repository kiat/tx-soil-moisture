# Met_Time_Series.md
The relevant code can be found in Met_Time_Series.ipynb(notebook/Met_Time_Series.ipynb)
### Shaojie Hou

Met_Time_Series processed the met data from six different stations in TX and there are
five steps for the construction of time series prediction model.

### Visualize the data of air temperature with timeseries
The unit timestamp of given data sets is hourly which means 24 rows of data within a day
among six stations. In order to visualize the time series correctly, we reconstructed
the range and unit length of x-axis and get the basic shape of air temperature in time series
for six different stations

### Build up STL model for temperature data of each station

STL model decomposes a time series into three components: trend, season(al) and residual. STL uses LOESS (locally estimated scatterplot smoothing) to extract smooths estimates of the three components.

In our data sets of six stations, based on the total time length of seven years, the period of STL models
are set as 7. By each sub elements of STL(), we received the 

        decomposition.plot()
        decomposition.seasonal.plot()
        decomposition.trend.plot()
        decomposition.residual.plot()

for each station. Based on the visualization of seasonality, we are able to find peaks and other main features of the given data.

These features and components combined in some way would be able to provide the regression model of observed time series and get prepared for the following steps.

### Test for staionary

Stationarity is important to assess how the data is perceived and predicted. Before forecasting or predicting the future, we need to ascertain that if each point is independent of one another, if it has no trend, if it exhibits constant variance over time, and if it has a constant autocorrelation structure over time.

To test the stationarity, we applied Augmented Dickey-Fuller Test and Order integration of one in Python. The null hypothesis H_0 of our test is assuming that The time series is non-stationary and the alternative hypothesis is H_A: The time series is stationary. Based on the statistic and p-value of adfuller, we determined the data from met stations are stationary.

### Visualize train and test set

Before the construction of prediction model, we split the data set into tranning and testing groups and visualize
them. The shaded area is the testing set and the remaining part is the training set.


### Forcasting Models

We use several forcasting models in the time series prediction:
Seasonal Autoregressive Integrated Moving Average Model, or SARIMA Model is the forcasting model we use for univariate time series data forecasting.

Rolling forecast uses historical data of met stations to predict future numbers continuously over the given period of time. 

There are two functions of prediction models. We also employ the Ljung-box test to determine if the test
result for correlation is different from zero. After the applying the prediction model over our datasets, we got the predicted value of air temperature among six different stations in TX.

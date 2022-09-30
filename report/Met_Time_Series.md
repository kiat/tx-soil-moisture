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

Stationarity is important to assess how the data is perceived and predicted. When forecasting or predicting the future, most time series models assume that each point is independent of one another. The test of stationarity is also the test of dependency among each individual data.

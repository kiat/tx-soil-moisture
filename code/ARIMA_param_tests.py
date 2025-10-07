from helpers.ARIMA_helpers import run_arima
from helpers.data_helpers import set_data
from helpers.data_helpers import retrieve_data
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

print("Starting Test")
target_idx = set_data("Station6", "SWC_5")
print("Data Collected")
target_feat = "SWC_5"
buff, train_df = retrieve_data()
test_df = retrieve_data(test = True)
end_test = test_df[[target_feat]]
end_train = train_df[[target_feat]]
data = end_train

print(len(end_test))

print("AUGMENTED DICKY FULLER" + "\n")
#DICKY FULLER
result = adfuller(data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))


#DIFFERENCING CHARTS
print("DIFFERENCING CHARTS" + "\n")
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
# Plot the original data
ax1.plot(data)
ax1.set_title('Original Series')
ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(data.diff())
ax2.set_title('1st Order Differencing')
ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(data.diff().diff())
ax3.set_title('2nd Order Differencing')
# Get the x-axis tick positions from the first graph (Original Series)
xticks = ax1.get_xticks()
# Add vertical lines at each x-axis mark within the data range
for x in xticks:
    if x >= 16436 and x <= len(data) - 1:
        ax1.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
        ax2.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
        ax3.axvline(x=x, color='red', linestyle='--', linewidth=0.8)

plt.tight_layout()
#plt.show()
plt.savefig('differencing_charts.png')
plt.show()

#ACF
print("ACF TEST Lag = 8761" + "\n")
fig, (ax1) = plt.subplots(1)
plot_acf(data["SWC_5"], ax =ax1, lags = 8761)
plt.savefig('ACF.png')
plt.show()

print("ACF TEST Lag = 3500" + "\n")
fig, (ax1) = plt.subplots(1)
plot_acf(data["SWC_5"], ax =ax1, lags = 3500)
plt.show()

print("ACF TEST Lag = 744" + "\n")
fig, (ax1) = plt.subplots(1)
plot_acf(data["SWC_5"], ax =ax1, lags = 744)
plt.show()

print("ACF TEST Lag = 96" + "\n")
fig, (ax1) = plt.subplots(1)
plot_acf(data["SWC_5"], ax =ax1, lags = 96)
plt.show()

#PACF
print("PACF TEST Lag = 96" + "\n")
fig, (ax1) = plt.subplots(1)
plot_pacf(data["SWC_5"], ax =ax1, lags = 96)
plt.savefig('PACF.png')
plt.show()

print("RUNNING ARIMA MODEL" + "\n")
arima = pm.ARIMA(order=(24*7*4, 0, 3), seasonal_order = (0,0,0,8760))
arima.fit(end_train, disp = 0)
forecast1 = arima.predict(len(end_test))
print("Model Ran" + "\n")
print(forecast1)

pred = pd.DataFrame(data = forecast1)
pred.to_csv("ARIMA_predictions_week.csv")
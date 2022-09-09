# Meteorological Data Preliminary Cleaning:

The relevant code can be found in [AirTemp-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/AirTemp-Comparisons-and-Cleaning.ipynb) , [RelHum-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/RelHum-Comparisons-and-Cleaning.ipynb) , [SolRad-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/SolRad-Comparisons-and-Cleaning.ipynb).

The meteorological data for each station contains several features:

Ppt: Precipitation; hourly rainfall total (mm)

Tair: air temperature (deg C)

RH: relative humidity (percentage)

Wind speed: meters/second

Wind direction: Degree

Srad: solar radiation (W/m^2)

In this preliminary cleanup, we visually compared the data of all six stations and used the mean to fill in gaps and replace abnormal values. Since the stations are approx. within 20 mi of each other, we can expect Tair, RH, and Srad values to be very similar between stations. Whereas, wind speed, wind direction and ppt can differ significantly. So, we focused on the Tair, RH, and Srad features to clean up using the mean of the six stations.

For each feature, we plotted all six station's data together and visually detected significant deviations of one station's data from the rest, as well as any gaps in the data on a year-to-year basis. Then, the station with gaps or anomalies was excluded from the mean calculation. The mean value of all other stations was then used to fill in gaps or replace the values of the anomalies.

## Air Temperature :

( code, graphs, and more details found in [AirTemp-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/AirTemp-Comparisons-and-Cleaning.ipynb) )

All Tair anomalies found were significant enough that we could write a function that replaces values below or above a certain limit with the mean. The limit was determined visually. Although recorded afterwards, the exact indices of the anomalies were not needed to be known to compute the mean and clean up the data.

For **2018** , anomalies and gaps in the data were found in Stations 1 and 5. Thus, Stations 1 and 5 were excluded from the Tair 2018 mean. The mean was then used to clean up 2018 Stations 1 and 5 data.

Station 1: Missing data from Apr 12 to May 7 and on Aug 2. Anomaly from May 6 to Aug 2

Station 5: Small gaps from Apr 5 to May 16. Small Anomalies from Apr 17 to May 13

For **2019** , anomalies were found in Station 1. Thus, Station1 was excluded from the Tair 2019 mean. The mean was then used to clean up 2019 Station 1 data.

Station 1: Anomaly from Mar 25 to Mar 26.

## Relative Humidity:

( code, graphs, and more details found in [RelHum-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/RelHum-Comparisons-and-Cleaning.ipynb) )

For this feature, all stations had an anomaly and/or gaps in the data and were scattered across several years. For example, 2018 RH had 3 different stations with anomalies. In this case, excluding 3 out of 6 stations from the 2018 mean would be ineffective. So here we determine the exact indices to change for each station first. Then we calculate a mean of the entire data timeframe that excludes the exact anomaly indices of each station. It's a more involved inspection compared to Tair clean up.

**Station 1:**

Missing data from 2018 Apr 12 to 2018 May 6

Anomaly from 2018 May 6 to 2018 Aug 2

**Station 2:**

Anomaly from 2015 Jul 18 to 2016 Jan 29

**Station 3:**

Anomaly from 2018 Aug 13 to 2019 Jul 11

Anomaly from 2019 Dec 21 to 2020 Mar 2

Anomaly from 2020 Sep 10 to 2020 Dec 31

Anomaly from 2021 Jan 21 to 2021 Apr 9

**Station 4:**

Anomaly from 2021 Jun 5 to 2021 Jun 8

**Station 5:**

Small gaps from 2018 Apr 5 to 2018 May 16

Small Anomalies from 2018 Apr 17 to 2018 May 13

Anomaly from 2019 Jan 14 to 2019 Dec 4

**Station 6:**

Small anomalies from 2020 Jul 2 to 2021 Feb 19

Small anomalies from 2021 Jun 23 to 2021 Aug 31

## Solar Radiation:

( code, graphs, and more details found in [SolRad-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/SolRad-Comparisons-and-Cleaning.ipynb))

In progress …

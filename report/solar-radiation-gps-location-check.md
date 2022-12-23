# Solar Radiation GPS Location Check

The relevant code can be found in [SolRad-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/SolRad-Comparisons-and-Cleaning.ipynb)

Here we focus on the solar radiation meteorological feature.

  Srad: solar radiation (W/m^2)

We expect solar radiation to be low during the night and high during the day. By knowing the times of sunrise and sunset, we can determine the Srad night time indices 
for each station. However, the exact timing of the sunrise and sunset changes by location and time. We can find the sunrise and sunset times using NOAA solar  calculations: https://gml.noaa.gov/grad/solcalc/. We can also use the Srad data to infer the night time indices. We can do so by choosing a minimun Srad value to be considered daytime and label any datapoint below that as nighttime.

### NOAA-Based Method

The NOAA calculations require a specific hour in time and are accurate to within a minute. Thus, NOAA calculations provides 24 different 
but very close sunset/sunrise times for each day. For our purposes, we use 12:00 noon time for each day to get one sunrise/sunset time answer.

The NOAA calculations also require the datetime to be converted into a Julian Day. The Julian day is commonly used in astronomical calculations and is the number of days
since the beginning of the Julian period, 4713 BC.


In [SolRad-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/SolRad-Comparisons-and-Cleaning.ipynb), the functions `get_julian_datetime()` and `get_sunlight_time()` perform these calculations.

`get_julian_datetime()` takes a date value as type `datetime.datetime` and returns the corresponding Julian Day.

`get_sunlight_time()` computes the sunrise and sunset times for each Julian Day using NOAA solar calculations.`get_sunlight_time()` takes in a station's longitude, latitude, time zone (UTC offset) and DateTimeIndex. The DateTimeIndex contains only the 12:00 noon time for each day in the data. The function outputs a dataframe with the sunrise time, sunset time, and sunlight duration (in hrs) for each day in the stations' data. Sunlight duration is calculated as the time between sunrise and sunset in hours.

### Data-Based Method

We can infer the sunrise/sunset time from the Srad data by considering points greater than a chosen limit as daylight hours for each day. We then take the first and last index of the daylight hours as sunrise and sunset, respectively.
In [SolRad-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/SolRad-Comparisons-and-Cleaning.ipynb), the function `get_sunlight_time_data()` performs this calculation.

`get_sunlight_time_data()` takes in the Srad data of one station and returns a dataframe with the sunrise time, sunset time, and sunlight duration (in hrs) for each day in the stations' data. The same format as `get_sunlight_time()`

### Checking data

With the NOAA sunrise/sunsets, can determine the night time indices of each station. In [SolRad-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/SolRad-Comparisons-and-Cleaning.ipynb), `get_night_data()` takes in the list of stations' NOAA sunlight results and the Srad data, and creates a dataframe containing the night time data of each station. Now we can take a closer look at the solar raditation data during the night and get statistics.

We can also compare the sunlight duration from the NOAA calculation to the sunlight duration from the Srad data. For each day, sunlight duration from data should be less than or equal to the sunlight duration from NOAA calculation due to weather afftecting solar radiation. Therefore, we can compute an accuracy for the sunlight duration.




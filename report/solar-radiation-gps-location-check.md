**In Progress...**

# Solar Radiation GPS Location Check

The relevant code can be found in [SolRad-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/SolRad-Comparisons-and-Cleaning.ipynb)

Here we focus on the solar radiation meteorological feature.
  Srad: solar radiation (W/m^2)

We expect solar radiation to be low during the night and high during the day. By knowing the times of sunrise and sunset, we can determine the Srad night time indices 
for each station. However, the exact timing of the sunrise and sunset changes by location and time. We can find the sunrise and sunset times using NOAA solar  calculations: https://gml.noaa.gov/grad/solcalc/.

The NOAA calculations require a specific hour in time and are accurate to within a minute. Thus, NOAA calculations provides 24 different 
but very close sunset/sunrise times for each day. For our purposes, we use 12:00 noon time for each day to get one sunrise/sunset time answer.

The NOAA calculations also require the datetime to be converted into a Julian Day. The Julian day is commonly used in astronomical calculations and is the number of days
since the beginning of the Julian period, 4713 BC.


In [SolRad-Comparisons-and-Cleaning.ipynb](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/SolRad-Comparisons-and-Cleaning.ipynb), the functions `get_julian_datetime()` and `get_sunlight_time()` perform these calculations.

`get_julian_datetime()` takes a date value as type `datetime.datetime` and returns the corresponding Julian Day.

`get_sunlight_time()` computes the sunrise and sunset times for each Julian Day using NOAA solar calculations.

`get_sunlight_time()` takes in a station's longitude, latitude, time zone (UTC offset) and DateTimeIndex. The DateTimeIndex contains only the 12:00 noon 
time for each day in the data. The function outputs a dataframe with the sunrise time, sunset time, and sunlight duration (in hrs) for each day in the stations' data.

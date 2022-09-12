**In Progress...**

By knowing the times of sunrise and sunset, we can determine the night time indices for each station. However, the exact timing of the sunrise and 
sunset changes by location and time. We can find the sunrise and sunset times using NOAA solar calculations: https://gml.noaa.gov/grad/solcalc/. 
The functions `get_julian_datetime()` and `get_sunlight_time()` perform these calculations.

The Julian day is commonly used in astronomical calculations. The Julian day is the number of days since the beginning of the Julian period, 4713 BC. 
`get_julian_datetime()` takes a date value as type `datetime.datetime` and returns the corresponding Julian Day.

`get_sunlight_time()` computes the sunrise and sunset times using NOAA solar calculations found here: https://gml.noaa.gov/grad/solcalc/. 
The NOAA calculations require a specific hour in time and are accurate to within a minute. Thus, NOAA calculations provides 24 different 
but very close sunset/sunrise times for each day. For our purposes, we only use the 12:00 noon time for each day to get one sunrise/sunset time answer. 

`get_sunlight_time()` takes in a station's longitude, latitude, time zone (UTC offset) and DateTimeIndex. The DateTimeIndex contains only the 12:00 noon 
time for each day in the data. The function outputs a dataframe with the sunrise time, sunset time, and sunlight duration (in hrs) for each day in the stations' data.

Shuhan Shen

ET0 Calculation

The term evapotranspiration (ET) is commonly used to describe two processes of water loss from land surface to atmosphere, evaporation and transpiration. We use ET0 to represent the evapotranspiration at surface. The FAO Penman-Monteith method to estimate ET0 can be derived by equation 6 at https://www.fao.org/3/x0490e/x0490e06.htm document, where:
ETo reference evapotranspiration [mm day-1],
Rn net radiation at the crop surface [MJ m-2 day-1],
G soil heat flux density [MJ m-2 day-1],
T mean daily air temperature at 2 m height [°C],
u2 wind speed at 2 m height [m s-1],
es saturation vapour pressure [kPa],
ea actual vapour pressure [kPa],
es - ea saturation vapour pressure deficit [kPa],
D slope vapour pressure curve [kPa °C-1].
A few of parameters we used from our cleaned datasets are includes:
Daily Average Windspeed in [m s-1], 
Daily Minimum and Maximum temperature in [°C]
Daily Average solar radiation [W/m^2]
Elevation above sea level in [m]
Daily Minimum and Maximum relevant humidity in [percentage]
Julian day 
Coordinate of the monitor station
I have give a test case at the beginning of code for you to check the answer. You can compare the result with Evapotranspiration calculator provided at Food and Agriculture Organization of the United Nations : https://www.fao.org/land-water/databases-and-software/eto-calculator/en/ (software downloads required)
You can also check the result with https://edis.ifas.ufl.edu/pdf/AE/AE45900.pdf step by step at the steps section in code. A few thing we still need to figure out are:
1.	What height does windspeed data from our stations are measured.
2.	Will color of water surface effect the result, what’s the parameter for color?(Professor mentions this during meetings)
3.	Do we need to around negative numbers to 0? most of negative numbers are around 0 now. 
4.	Make sure replace the elevation number to accurate number at monitor station locations. Now default is at location of Austin.
The intuition of finding ET0 is to understand how does evaporation effect soil moisture (SWC_). Currently I did the data visualization to compare SW_5 data at Station 2 with ET0. For further analysis, you might also check SWC_10, SWC_20 and SWC_50, also at other stations. The hypothesis we have is evapotranspiration has positive correlation with soil moisture, if the trend doesn’t match might result by rainfall. Later, Tylor will combine rainfall data to make more in-depth observation. 

A few modifications I made after my last day meeting are:
1.	I change the dataset from raw data to cleaned data
2.	I have fixed the SWC_5 data, now the data correctly show data is percentage
3.	The count of every features in the concat DataFrame DF2 are in consistency, The issue previously is because I used left join, so using inner join instead.



NOAA calculation 

NOAA is an official site of provides a calculation for the general solar position. The calculation is composed of two functions: get_julian_datetime() and get_sunlight_time(), the first one is calculating the corresponding Julian time, which gives the fraction of the year. Next, in the second function, we are using Julian time to estimate the equation of time (in minutes) and the solar declination angle (in radians). Then we transfer time offset to the true solar time in minutes. The degree between the horizon and height of the sun is called the solar zenith angel, which also leads to hour angel (ha). In this phase, we require to have the coordination of the monitor stations and local time zone as parameters, because location relates to the solar position. From hour angel, we can calculate sunrise/sunset time directly, where the positive number corresponds to sunrise, negative to sunset. lastly, the daytime duration is calculated by the subtraction of sunset time and sunrise time. And be aware, that since the station monitors every day's data, the NOAA calculation gives data for every hour. In order to keep it consistent, we decide to use 12 pm in noon to represent the day from NOAA calculation. 
Once we have the result from the NOAA calculation method, we want to use it to verify solar radiation data that we are monitoring from stations, the comparison will be visualized in the next step


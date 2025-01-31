

SM_MET_TX.zip includes: 

# soil_station folder:

SM_1.dat to SM_6.dat includes 6 different stations at different locations (latitude and longitude). Each station has 4 sensors in soil at different depths. We measure soil water content and soil 
temperature. Each .dat file includes the following columns:

**Date:** Synchronized time stamps (CST)

**Ppt:** Precipitation; hourly rainfall total (mm),


**Soil Moisture** is a dimensionless property and has no unit. It is volume divided by volume (we can say the unit is cubic meter divided by cubic meter or m^3/m^3). The value can be from 0.0 to 0.6 practically but in theory it can be between 0.0 to 1.0 


**SWC_5:** Soil water content or soil moisture at 5 cm depth, 

**SWC_10:** Soil water content or soil moisture at 10 cm depth, 

**SWC_20:** Soil water content or soil moisture at 20 cm depth, 

**SWC_50:**  Soil water content or soil moisture at 50 cm depth,     

**T_5:** Soil Temperature at 5 cm depth,

**T_10:** Soil Temperature at 10 cm depth,

**T_20:** Soil Temperature at 20 cm depth,

**T_50:** Soil Temperature at 50 cm depth,

**Flag:** is an integer that we assign to each data point based on the quality control protocols and physics of the evapotranspiration process, we don't use it in the ML or we can use ML to predict it latter. 




# met_station folder:

MET_1.dat to MET_6.dat includes 6 different weather stations at different locations (latitude and longitude). Weather Station MET_1.dat is located at the same place that soil station SM_1.dat is located. 

This is correct for other stations, i.e. station SM_i and MET_i are in the same location. The "Date" column in MET_i station starts from "10/01/14 00:00" this is just because weather stations have been installed bofre soil stations, we only consider data starting from  "01/01/15 00:00" onward,  the date that Soil moisture data starts. Each station has different sensors/instruments. 

We measure air temperature, humidity, wind speed and direction, and solar radiation. Each .dat file includes the following columns:

**Date:** Synchronized time stamps (CST)

**Tair:** air temperature (deg C)

* Local record low≥ Temp≤ local record high, 
* Temp ≤ 5°C from previous hourly record, 
* Temp varies ≥ 0.5°C over 12 consecutive hours, or per site specific climatology criteria


**RH:** relative humidity (percentage): https://en.wikipedia.org/wiki/Humidity

* Dew Pont Temp ≤ Ambient temp for time period,
* Dew Pont Temp < 5°C change from previous hour,
* Dew Pont Temp≥ 0.5°C from previous hour, and
* Dew Pont Temp < Ambient Temp for 12 consecutive hours.



**Wind speed:** meters/second

* 0 m/s ≥ WS ≤ 25 m/s, 
* WS varies ≥ 0.1 m/s for 3 consecutive hours, 
* WS varies ≥ 0.5 m/s for 12 consecutive hours, 
* or per site specific climatology criteria
* 1/week or more frequent 

**Wind direction:** Degree

* 0°≥ WD ≤ 360°, 
* WD varies ≥ 1°/3 consecutive hours, 
* or per site specific climatology criteria


**Srad:** solar radiation (W/m^2) https://en.wikipedia.org/wiki/Solar_irradiance

* Night time SR = 0,
* Day time SR < max SR for date and latitude



These data are a little different from Houston data. But, soil moisture, precipitation, humidity, wind speed and direction and solar radiation are common parameters that we use. 

I think, something similar to this: https://github.com/lpphd/multivariate-attention-tcn, would be interesting to build and add, so that in real time we can predict each station soil moisture for the next day and since we get data each hour, we can check the ML model every hour.


# GPS Locations

1-	Latitude	30.3989
Longitude	-98.6105
Logger ID	CR1000-5

2-	Latitude	30.4193
Longitude	-98.8046
Logger ID	CR200-26 (CR1000-1)

3-	Latitude	30.4421
Longitude	-98.8427
Logger ID	CR1000-6

4-	Latitude	30.4600
Longitude	-98.9407
Logger ID	CR1000-4

5-	Latitude	30.2454
Longitude	-98.7059
Logger ID	CR1000-2

6-	Latitude	30.2758
Longitude	-98.7242
Logger ID	CR1000-3


More information here 

https://www.beg.utexas.edu/research/programs/txson 




# Data Imputation 

The first step for imutation was to visualize the data and to document what time ranges were missing for each of variable of each stations' data in the time series. After plotting the data, these were the gaps we were able to identify.

Soil station number 1
* Start: 1 January 2015
* Stop: 1 October 2021
* Missing Data:
    * April-may 2018 missing


Soil station number 2
* Start: 1 January 2015
* Stop: 1 October 2021
* Missing Data:
    * No data after some point in June 2021

Soil station number 3
* Start: 1 January 2015
* Stop: 1 October 2021
* Missing Data:
    * No missing data


Soil station number 4
* Start: 1 January 2015
* Stop: 1 October 2021
* Gaps:
    * No SWC50/T50 at all

Soil station number 5
* Start: 1 January 2015
* Stop: 1 October 2021
* Gaps:
    * No SWC20 until March of 2016
    * No good T20 data until of March of 2016
    * Gaps in moisture data in April/May 2018


Soil station number 6
* Start: 1 January 2015
* Stop: 1 October 2021
* Gaps:
    * No missing data


Weather station number 1
* Start: 1 October 2014
* Stop: 1 September 2021
* Missing Data:
    * Data missing in April-May of 2018
    * No data After 1 September 2021

Weather station number 2
* Start: 2 October 2014
* Stop: 6 June 2021
* Missing Data:
    * RH data missing from July 2015-February 2016
    * No data after 6 June 2021


Weather station number 3
* Start: 12 November 2014
* Stop: 1 September 2021
* Missing Data:
    * No missing data seen


Weather station number 4
* Start: 13 November 2014
* Stop: 1 September 2021
* Missing Data:
    * No missing data seen
	

Weather station number 5
* Start: 12 November 2014
* Stop: 1 September 2021
* Missing Data:
    * Strange data from April-May 2018
    * Very strange RH data in 2019
    * Strange Tair data in February 2021

Weather station number 6
* Start: 3 December 2014
* Stop: 1 September 2021
* Missing Data:
    * Strange data from April-June 2018
    * Very strange RH data in 2019  


Next we tried filling in the data using several different imputation methods as you can see in the different functions of this [notebook](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/soil_moisture_imputation_methods.ipynb)

More details about the application of these different methods can be found at these resources:
* [https://www.section.io/engineering-education/missing-values-in-time-series/](https://www.section.io/engineering-education/missing-values-in-time-series/)
* [https://www.geeksforgeeks.org/how-to-calculate-moving-average-in-a-pandas-dataframe/](https://www.geeksforgeeks.org/how-to-calculate-moving-average-in-a-pandas-dataframe/)

After visualizing the filled in data for each of the different imputation methods, we identified that data that was missing at a specific point in time  for a station would best be filled by using the mean of data values from the other stations also at that specific point in time. This resulted in the most realistic looking plots for the filled in data and so our hypothesis is that this will yield the best imputation for training a machine learning model.

We then combined our efforts in a single [notebook](https://github.com/kiat/tx-soil-moisture/blob/main/notebook/Soil_Cleaning.ipynb) that runs the anomaly detection code and the imputation code and outputs new csv data files with the cleaned up data for use in the model.

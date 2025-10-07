This document will outline the processes used to create the Raw_Merged, Cleaned_Merged and Simulate_Cleaned_Merged datasets.  

The Jupyter Notebook used to create these datasets can be found here: 

Overview: 

To begin, we have data from 6 weather and soil moisture stations across Texas. Each station has two data frames: one for soil moisture content and  ground temperature, and one for meteorological data.  All the data is from between 2014 and 2021. 

Raw_Merged:

  Both the soil moisture and meteorological datasets are read in. 
  
  All values are converted from strings to numerics. 
  
  For each station, we merge their soil moisture and meteorological data frames into one data frame, first taking the intersection of their indexes. 
  
  Then we store the data frames as a csv file for each station.

Cleaned_Merged:

  Using the raw merged data frames, 
  
  Remove the Flag feature from each station. 
  
  Forward fill each data frame with a limit of 48 hours. So any value that is NaN will be set to the value before it so long as there is an observed value within 48 hours. 
  
  Then all indexes for which no features have any non-NaN are dropped. 
  
  Then we add the longitude and latitude for each station.
  
  Then we store the data frames as a csv file for each station.

Simulate_Cleaned_Merged: 
  
  At this point, we have large numbers of missing values for T_50 , SWC_50 and SWC_20 features. 
  
  For T_50 and SWC_50, all the missing values occur for Station 4 where no T_50 or SWC_50 values are present.
  
  For SWC_20 the majority of the missing values are from Station 5 and a few are from Station 1.
  
  For all these missing values for a feature, we train a linear model to predict that feature on all of the indexes from every station where no values are missing for any feature. 
  
  We then use these linear models to predict the missing values. 
  
  Then we store the data frames as a csv file for each station.

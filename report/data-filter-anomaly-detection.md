# Data Filter and Anomaly Detection 
### Shaojie Hou

There are three functions get involved in the data filtering of weather datasets, including 'tair_dete', 'windspeed_detec', and 'winddir_detec' and two fucntions to actualize visualizations for datasets before and after filterings.

### Data Filtering
For data filtering functions, each of them is required to input a list including datasets and a list of column names. Then those functions will filter the datasets based on proper data ranges based on its specific column. Filtering functions will then modify dataframes with nan for future modifications.

### Data Visualization
For data visualizations of filtering functions, one of them is used before filtering and the remaining is used for after. With the comparison of graphs before and after filtering, we are able to see if the data has been successfully cleaned up with filtering functions.

There are also functions used for outliers detections but haven't been used on this stage. Those functions utilized Moving Average Outlier Detection on a monthly basis to detect potential outliers in input dataframes and replace them with nan for further use. 


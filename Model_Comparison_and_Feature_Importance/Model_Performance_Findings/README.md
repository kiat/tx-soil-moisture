# Soil Moisture Sensor Model Performance and Feature Importance - Texas soil moisture dataset


TODOs: 

* Combine Alex and I's code into a single codebase that can be adjusted across stations and different parameters
* Compare Model Performance
* 2 Methods of Testing Feature Importance


# Comparing Model Performance

Configure codebase to test on all stations and use 5 years for training and final year for test data.
Test model performances across different configurations regarding label features, input window size, output window size, and shift amount.
Label Features - SWC_5, SWC_10, and SWC_20.
Input Window Sizes - 24 Hours, 48 Hours, 168 Hours, 240 Hours
Output Window Sizes - 1 Hour, 6 Hours, 12 Hours, 24 Hours, 48 Hours, 168 Hours

# Methods of Testing Feature Importance

## 1. Drop Features One at a Time

Testing across various models and label features.
Done by dropping a singular feature at a time and documenting change in metrics compared to Models with all features included.

## 2. Testing One Feature at a Time and Combining

Once again, testing across various models and label features.
Will be done by testing one feature at a time and documenting metrics.
Top 2 Features will be selected to create a model together. Process repeated until all combinations of 2 features tested.
Top 3 Features will be selected to create a model together. Repeat until all combinations of 3 features tested.
Build a final model using the highest performing features and configuration.

# CYGNESS 

https://cygnss.engin.umich.edu/


# SMAP Satellite 

It is better to use SMAP Satellite 

https://smap.jpl.nasa.gov/


# Merge.py & Modify.py

These scripts are responsible for data integration and modification:

Merge.py:
- Combines AMSR data with SMAP satellite data
- Integrates the satellite data with simulated data
- Ensures proper alignment of timestamps and data points across different sources
- Handles any discrepancies in data formats or units

Modify.py:
- Performs necessary modifications on the merged dataset
- May include data cleaning, formatting, or additional preprocessing steps
- Prepares the data for further analysis and modeling

Together, these scripts create a comprehensive dataset that combines multiple sources of soil moisture data, providing a robust foundation for subsequent analysis and modeling tasks.

MetMerge.py:
- Combines AMSR data with SMAP satellite data
- Keeps out the SWC data

# Cleaning_and_Analysis.ipynb

This Jupyter notebook focuses on data cleaning and analysis for soil moisture and meteorological data. Key steps include:

- Importing necessary libraries (pandas, numpy, matplotlib, etc.)
- Reading and preprocessing data from multiple stations
- Merging soil moisture and meteorological data
- Cleaning the data by removing spaces in column names, converting data types, and handling missing values
- Creating visualizations to explore the data
- Performing feature engineering, such as calculating moving averages
- Normalizing the data
- Saving the cleaned and processed data to CSV files

# Data_Exploration.ipynb

This notebook is dedicated to exploring the cleaned and merged data. It includes:

- Loading the preprocessed data
- Generating descriptive statistics
- Creating various visualizations to understand the relationships between different features
- Analyzing correlations between variables
- Exploring time series patterns in the data

# model_comparison.py

This Python script focuses on evaluating feature importance for different models, particularly RNN and CNN. Key components include:

- Importing required libraries and modules
- Loading and preprocessing the data
- Defining functions for feature importance calculation
- Implementing RNN and CNN models for feature importance evaluation
- Calculating and displaying feature importance scores for different metrics (MSE, MAE, MAE%, R²)
- Handling exceptions and errors in the evaluation process
- Saving results to CSV files


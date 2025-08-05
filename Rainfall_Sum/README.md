# Overview
We ran numerous hypothesis tests to determine things like seasonality on rainfall, air temperature, and soil moisture. We mainly focused on rainfall amount, with air temperature and soil moisture being more of a subtask. Evan Nguyen was the team lead and focused on some of the hypothesis tests for rainfall and applying the tests to air temperature and soil moisture. Abi Vijayan and Selim Gurkas worked on the hypothesis tests for rainfall. Specifically, ANOVA and Kriging analysis respectively. Tiffany Nguyen focused on converting the visuals into PDF. Leo Wang focused on converting our code into a report on Overleaf.

## Tests
* Kruskal-Wallis: It assesses whether samples originate from the same distribution, without assuming normality.
* ANOVA:
* Two-Sample T-Test:
* Mann-Kendall Trend Test: non-parametric method used to identify monotonic trends (consistently increasing or decreasing) in a time series without assuming any particular distribution.
* OLS Regression: 
* Autocorrelation Function: ACF helps identify whether rainfall patterns repeat at regular intervals.
* Kriging Analysis:

# Assumptions
We used data from Revised_Final_Data. We assumed the data was properly cleaned, but Station 1 had some missing data. We decided to remove Station 1 from our tests.

## Usage
Load datasets from multiple stations using the load_rainfall_data() function.
Choose to load the data in by year, month, or month_matrix.
Run the hypothesis tests and output results.

# Results
The results of our tests on rainfall are located on an Overleaf document linked below:
[CS Project on Rainfall](https://www.overleaf.com/project/686417ad4f0d44606ec54a89)

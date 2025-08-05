# Overview
We ran numerous hypothesis tests to determine things like seasonality on rainfall, air temperature, and soil moisture. We mainly focused on rainfall amount, with air temperature and soil moisture being more of a subtask. Evan Nguyen was the team lead and focused on some of the hypothesis tests for rainfall and applying the tests to air temperature and soil moisture. Abi Vijayan and Selim Gurkas worked on the hypothesis tests for rainfall. Specifically, ANOVA and Kriging analysis respectively. Tiffany Nguyen focused on converting the visuals into PDF. Leo Wang focused on converting our code into a report on Overleaf.

## Tests
*Kruskal-Wallis: A non-parametric test used to determine if there are statistically significant differences between the medians of three or more independent groups. It does not assume a normal distribution.

*ANOVA (Analysis of Variance): A parametric test that compares the means of three or more groups to determine if at least one group mean is significantly different from the others. It assumes normality and equal variances.

*Two-Sample T-Test: A parametric test used to compare the means of two independent groups to determine if they are significantly different from each other. Assumes normality and equal variance.

*Mann-Kendall Trend Test: A non-parametric test used to detect monotonic (increasing or decreasing) trends in a time series without assuming any specific distribution. Often used in environmental data analysis.

*OLS Regression (Ordinary Least Squares): A method for estimating the linear relationship between a dependent variable and one or more independent variables. Assumes linearity, independence, homoscedasticity, and normally distributed errors.

*Autocorrelation Function (ACF): A statistical tool used to measure the correlation of a time series with its own past values (lags). Helps identify repeating patterns or seasonality in data, such as rainfall or temperature cycles.

*Kriging Analysis: A geostatistical interpolation method used to estimate values at unmeasured locations based on spatial autocorrelation. Commonly used in environmental and geospatial analysis to create continuous surface maps from discrete data points.

# Assumptions
We used data from Revised_Final_Data. We assumed the data was properly cleaned, but Station 1 had some missing data. We decided to remove Station 1 from our tests.

## Usage
Load datasets from multiple stations using the load_rainfall_data() function.
Choose to load the data in by year, month, or month_matrix.
Run the hypothesis tests and output results.

# Results
The results of our tests on rainfall are located on an Overleaf document linked below:
[CS Project on Rainfall](https://www.overleaf.com/project/686417ad4f0d44606ec54a89)

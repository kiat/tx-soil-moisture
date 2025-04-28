# Interpolation Strategy for Missing Data

We are imputing missing data based on the length for each missing data:

## Short Gaps (Daily missing)
- Method: 
  - KNN (k-nearest neighbors from scikit-learn)
    - Non-parametric supervised learning algorithm
    - Predicts based on the nearest values beside it (so works well with short term gaps)
    - Use 24 hours before and after the missing point - KNN
- Variables:
  - SWC, SWC_5-50: KNN or linear interpolation
  - Tair: use nearby station values
  - RH: has an inverse relationship with Tair, use this knowledge to check consistency
  - Wind speed: not sure
  - Wind direction: circular interpolation, because circular data
  - Srad: 24 nearest neighbors before and after (KNN)

## Medium Gaps (1 week or less missing)
- Methods:
  - SARIMA/ARIMA: accounts for seasonality
  - Regression-based models: Random Forest, Gradient Boosting
    - Can use unless all features are NaN
  - NEW --> Holt-Winters: 
    - Good for seasonality, cyclic behavior, and smoothing averages
- Variables:
  - Tair: nearby station values
  - RH: same as short gaps (inverse with Tair)
  - Wind speed: not sure
  - Wind direction: circular interpolation because we have circular data
  - Srad: not sure
  - Ppt: 
    - Use nearby station values
    - Idea: classify into "yes it rained" and "no it didn’t rain"
    - Only if "yes" predict the rain amount

## Long Gaps (1 week - 1 month missing)
- Methods:
  - Prophet (https://facebook.github.io/prophet/) (from Facebook/Meta)
    - Designed for time series with strong seasonal effects
    - Robust to missing data and seasonality changes
  - Explore state-space
- Variables --> similar to as medium gaps

## Very Long Gaps (more than 1 month missing)
- Methods:
  - ML algorithms:
    - Random Forest
    - Gradient Boosting
    - LSTM (Long Short-Term Memory neural networks)

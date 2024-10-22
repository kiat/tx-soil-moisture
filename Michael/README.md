Models Used
We compare a range of machine learning and deep learning models for predicting soil moisture content over different time intervals. Each model is trained and evaluated across multiple configurations, varying the input and output time steps.

Models
Baseline: A simple baseline model that predicts the last observed SWC value for the given forecast window.
Multi-step Linear: A dense layer model that predicts the future steps based on the last time step of the input.
Multi-step Dense: A fully connected neural network with 512 hidden units, allowing more complex relationships between inputs and outputs.
Simple RNN: A basic recurrent neural network that handles sequential data, capturing temporal patterns.
CNN: A convolutional neural network that applies filters to recent input data to capture local patterns.
LSTM: A Long Short-Term Memory network, which is effective at capturing long-term dependencies in sequential data.
Autoregressive: A model that forecasts based on past observations and global average pooling.
Bi-directional LSTM: An advanced LSTM model that looks at both past and future context by processing sequences in both directions.
For each model, we evaluate its performance in terms of mean absolute error (MAE), mean squared error (MSE), and mean absolute percentage error (MAPE).

Model Evaluation
The models are trained and evaluated on different configurations, where we vary the:

Input steps: The number of previous time steps used as input.
Output steps: The number of future time steps we want to predict.
Configurations range from short-term predictions (1 hour) to long-term forecasts (48 hours).

Example Configurations:
Input steps: 24 (one day), Output steps: 1, 6, 12, 24, or 48 hours.
Features: Soil moisture content at different depths (SWC_5, SWC_10, SWC_20, SWC_50).
Each model is trained, validated, and tested on these configurations, and performance metrics are logged to track which models perform best under different settings.

Feature Importance Determination
We also assess feature importance by measuring the impact of dropping specific features from the model and evaluating how it affects performance. This is done by:

Calculating Original Performance: The models are first trained with all available features.
Dropping Individual Features: One feature is removed at a time, and the model is retrained. The change in MAE after dropping the feature gives us an indication of its importance.
For example, we test the importance of meteorological features like temperature at different depths (T_5, T_10, etc.) by removing them and observing how it impacts soil moisture prediction.

Example Process:
Target Feature: SWC_5 (Soil moisture content at 5cm depth).
Other Features Dropped: Temperature at various depths (T_5, T_10, T_20, T_50) and soil moisture content at different depths (SWC_10, SWC_20, SWC_50).
The results are logged in CSV files to track how each feature impacts the accuracy of different models.

Usage
Load datasets from multiple stations using the load_all_data() function.
Preprocess and normalize the data for training.
Train the models using the train_and_evaluate_models() function.
Use drop_feature_and_evaluate() to determine feature importance and log the results.
Results
The results of the model comparisons and feature importance evaluations are saved in CSV files:

model_results.csv: Contains the performance metrics (MAE, MSE, MAPE) for each model and configuration.
feature_importance_results.csv: Logs the change in MAE when specific features are dropped from the model.

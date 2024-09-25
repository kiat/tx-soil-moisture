from helpers.data_helpers import set_data, read_data, get_data
from helpers.model_helpers import train_model, evaluate_model, evaluate_all
from helpers.ARIMA_helpers import run_arima
from runner import run_model
from models import model_dict
import tensorflow as tf
import os
import shap
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# Define shape_dict, station_list, swc_list
shape_dict = {
    "s1": [24*7, 24, 1024],  # You can add more shapes if needed
}

station_list = ["Station1"]
swc_list = ["SWC_5"]

# Experiment loop
trial_num = 1
for key, shape in shape_dict.items():
    for station in station_list:
        for swc in swc_list:
            trial = "Trial" + str(trial_num)
            print("Running Experiment - Key: ", key, " Shape: ", shape, "  Station: ", station, "  SWC:", swc)
            # Run the model (assumes run_model trains and returns the model)
            model = run_model(model_dict(shape=shape), trial, remove_met=False, test_station=station, target_col=swc, max_epochs=50, patience=5, trial_shape=shape)
            trial_num += 1


#SHAP
def calculate_shap_importance(model, training_data_sample, test_data_sample):
    explainer = shap.DeepExplainer(model, training_data_sample)  # Explain predictions
    shap_values = explainer.shap_values(test_data_sample)

    # plot shap values
    shap.summary_plot(shap_values, test_data_sample)
    return shap_values

# now looking at importance of permutation
def calculate_permutation_importance(model, test_data, test_labels):
   
    y_pred = model.predict(test_data)
    
    #  get permutation importance
    results = permutation_importance(model, test_data, test_labels, scoring='neg_mean_squared_error')
    
    
    #  scores
    importance_df = pd.DataFrame(results.importances_mean, index=test_data.columns, columns=["Importance"])
    print(importance_df.sort_values(by="Importance", ascending=False))
    return importance_df


training_data_sample = np.random.rand(100, 1024)  
test_data_sample = np.random.rand(10, 1024)  

# Calculate SHAP importance
shap_values = calculate_shap_importance(model, training_data_sample, test_data_sample)



# hello 
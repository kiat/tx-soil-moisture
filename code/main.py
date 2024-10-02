from helpers.data_helpers import set_data, read_data, get_data
from helpers.model_helpers import train_model, evaluate_model, evaluate_all
from helpers.ARIMA_helpers import run_arima
from runner import run_model
from models import model_dict
import tensorflow as tf
from helpers.feedback_model import FeedBack
import os


shape_dict = {
    "s1": [24*7, 24, 1024], 
    "s2": [24*7, 24*7, 1024],
}


station_list = ["Station1","Station6"]

swc_list = ["SWC_5","SWC_10","SWC_20","SWC_50"]


trial_num = 1
for key, shape in shape_dict.items():
    for station in station_list:
        for swc in swc_list:
            trial = "Trial" + str(trial_num)
            print("Running Experiment - Key: ", key," Shape: " , shape , "  Station: ",  station, "  SWC:" , swc)
            run_model(model_dict(shape = shape), trial, remove_met = False, test_station = station, target_col = swc, max_epochs = 50, patience = 5, trial_shape = shape)
            trial_num += 1


run_arima(target_feat = "SWC_5", exo_feats = ["SWC_10","SWC_20","SWC_50"], seasonal = False, stepwise = False)


'''
for station, df in dfs.items():
    print(df.isnull().sum())
    print(df.shape)
'''


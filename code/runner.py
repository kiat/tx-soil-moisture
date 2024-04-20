from helpers.data_helpers import set_data
from helpers.model_helpers import train_model, evaluate_all
# from helpers.ARIMA_helpers import add_baseline
import os

def run_model(model_dict, folder_name, remove_met = False, test_station = "Station6", target_col = "SWC_5", max_epochs = 50, patience = 10, trial_shape = [21, 7, 2048]):
    os.mkdir(folder_name)
    target_idx = set_data(test_station, target_col, remove_met)
    window_size = trial_shape[0]
    shift_amt = trial_shape[1]
    batch_size = trial_shape[2]

    train_model(model_dict, patience, folder_name, max_epochs, window_size, shift_amt, target_idx, batch_size)
    #evaluate_model(save_path, window_size, shift_amt, target_idx, batch_size)
    evaluate_all(folder_name, model_dict, window_size, shift_amt, target_idx, batch_size)
    #fix baseline needs to have length cut
    #add_baseline(folder_name, target_feat = target_col, exo_feats = ["T_5", "Ppt_y","SWC_50"], SARIMAX = False) 


from helpers.data_helpers import set_data
from helpers.model_helpers import train_model, evaluate_model, evaluate_all
from runner import run_model
from models import model_dict
import tensorflow as tf
from helpers.feedback_model import FeedBack
import os


print("SWC_5 START" + "\n")
run_model(model_dict(), "TrialB1", remove_met = False, test_station = "Station1", target_col = "SWC_5", max_epochs = 30, patience = 5, trial_shape = [24*7, 10, 512])
run_model(model_dict(), "TrialB2", remove_met = False, test_station = "Station1", target_col = "SWC_5", max_epochs = 30, patience = 5, trial_shape = [24*7, 24, 512])
run_model(model_dict(), "TrialB3", remove_met = False, test_station = "Station1", target_col = "SWC_5", max_epochs = 30, patience = 5, trial_shape = [24*7*3, 24*7, 512])
run_model(model_dict(), "TrialB4", remove_met = False, test_station = "Station1", target_col = "SWC_5", max_epochs = 30, patience = 5, trial_shape = [24*7*8, 24*7, 512])

print("SWC_50 START" + "\n")
run_model(model_dict(), "TrialC1", remove_met = False, test_station = "Station1", target_col = "SWC_50", max_epochs = 30, patience = 5, trial_shape = [24*7, 10, 512])
run_model(model_dict(), "TrialC2", remove_met = False, test_station = "Station1", target_col = "SWC_50", max_epochs = 30, patience = 5, trial_shape = [24*7, 24, 512])
run_model(model_dict(), "TrialC3", remove_met = False, test_station = "Station1", target_col = "SWC_50", max_epochs = 30, patience = 5, trial_shape = [24*7*3, 24*7, 512])
run_model(model_dict(), "TrialC4", remove_met = False, test_station = "Station1", target_col = "SWC_50", max_epochs = 30, patience = 5, trial_shape = [24*7*8, 24*7, 512])








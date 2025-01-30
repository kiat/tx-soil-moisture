# model_manager.py
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

def save_model(model, base_filename="my_model", folder="saved_models"):
    """
    Saves a Keras model in HDF5 format with a timestamped filename.
    Example: saved_models/my_model_20230905_122030.h5
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{base_filename}_{timestamp}.h5"
    filepath = os.path.join(folder, filename)
    model.save(filepath)
    print(f"Model saved at: {filepath}")
    return filepath

def load_model(filepath):
    """
    Loads a Keras model from an HDF5 file or SavedModel directory.
    """
    model = keras_load_model(filepath)
    print(f"Model loaded from: {filepath}")
    return model

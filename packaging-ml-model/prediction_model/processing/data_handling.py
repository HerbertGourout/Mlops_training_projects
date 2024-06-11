
import numpy as np
import os
import pandas as pd
import joblib
import sys

root_package = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(root_package))
from prediction_model.config import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.datapath,file_name)
    data = pd.read_csv(filepath)
    return data

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.save_model_path,config.save_model)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.save_model}")

#Deserialization
def load_pipeline():
    save_path = os.path.join(config.save_model_path,config.save_model)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded

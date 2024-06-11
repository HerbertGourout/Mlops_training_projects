import pandas as pd
import numpy as np 
from pathlib import Path
import os
import sys
root= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(root))
from prediction_model.processing.data_handling import load_dataset, save_pipeline
from prediction_model.processing.dataprocessing  import ModelTrainer
from prediction_model.config import config
import prediction_model.pipeline as pipe
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")




# Initialiser et exécuter le processus
def perform_train():
    train = load_dataset(config.train_file)
    #test = load_dataset(config.test_file)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train.drop(columns=['Exited']),
                                                train['Exited'],
                                                test_size=0.2,
                                                random_state=0,
                                                stratify=train['Exited'])

    model_trainer = ModelTrainer(pipe.pipeline, Xtrain, Ytrain, Xtest, Ytest)
    model_trainer.run()


if __name__=='__main__':
    perform_train()


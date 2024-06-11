import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys
root= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(root))

from prediction_model.config import config  
from prediction_model.processing.data_handling import load_pipeline,load_dataset
classification_pipeline = load_pipeline()
print(classification_pipeline)
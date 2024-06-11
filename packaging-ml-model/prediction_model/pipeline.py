import sys
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
root= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(root))
from prediction_model.config import config
import prediction_model.processing.dataprocessing  as pp

# Exemple de pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('drop_columns', pp.DropColumns(columns=["id", "Surname", "CustomerId"]), config.columns_drop),
        ('fill_missing', pp.FillMissing(target='Exited'), []),
        ('handle_rare_categories', pp.HandleRareCategories(col_list=config.cols_rare), []),
        ('one_hot_encoding', pp.OneHotEncoderCustom(cols_to_encode=config.cat_cols_to_encode), config.cat_cols_to_encode)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])
print("Hello word")

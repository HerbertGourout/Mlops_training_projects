from pathlib import Path
import os
import sys
# Ajout du chemin du répertoire parent au chemin de recherche des modules
root_package = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(root_package))


datapath = os.path.join(root_package,"prediction_model/datasets")
train_file = "train.csv"
test_file  = "test.csv"
save_model = "classification.pkl"
save_model_path = os.path.join(root_package,"prediction_model/trained_models")
target = 'Exited'
# Les colonnes à transformer
cont_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
cat_cols_to_encode = ["Geography", "Tenure", "NumOfProducts", "Gender"]
cols_rare = ["Tenure", "NumOfProducts"]
columns_drop =["id", "Surname", "CustomerId"]

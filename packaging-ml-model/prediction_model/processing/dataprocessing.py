
import numpy as np
import pandas as pd
import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LogisticRegression
root= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(root))
import prediction_model.processing.data_handling  as dh
from sklearn.metrics import accuracy_score

class DropColumns(BaseEstimator, TransformerMixin):
    """A transformer class to drop specified columns from a DataFrame.

    Parameters:
    -----------
    columns : list
        List of column names to be dropped.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data.

    transform(X):
        Transform the data by dropping the specified columns.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.

    Returns:
    --------
    DataFrame
        The transformed DataFrame with specified columns dropped.
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        return [col for col in input_features if col not in self.columns]
# DropColumns Transformer
class FillMissing(BaseEstimator, TransformerMixin):
    def __init__(self, target, max_iterations=10):
        self.target = target
        self.max_iterations = max_iterations
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        train = X.copy()
        if self.target in train.columns:
            train = train.drop(columns=self.target)
        
        df = pd.concat([train, y], axis="rows").reset_index(drop=True)
        numeric_features = df.select_dtypes(include=np.number).columns
        categorical_features = df.select_dtypes(include='object').columns

        if len(numeric_features) > 0:
            imputer_numeric = KNNImputer()
            df[numeric_features] = imputer_numeric.fit_transform(df[numeric_features])
        
        if len(categorical_features) > 0:
            for feature in categorical_features:
                mode_value = df[feature].mode()[0]
                df[feature] = df[feature].fillna(mode_value)
        
        return df.iloc[:train.shape[0]].reset_index(drop=True)

    def get_feature_names_out(self, input_features=None):
        return input_features
    
    # HandleRareCategories Transformer
class HandleRareCategories(BaseEstimator, TransformerMixin):
    def __init__(self, col_list):
        self.col_list = col_list
    
    def fit(self, X, y=None):
        self.common_values = {}
        for col in self.col_list:
            self.common_values[col] = set(X[col].value_counts().index)
        return self
    
    def transform(self, X):
        df = X.copy()
        for col in self.col_list:
            common = self.common_values[col]
            df[col] = df[col].apply(lambda x: nearest_val(x, common) if x not in common else x)
        return df

    def get_feature_names_out(self, input_features=None):
        return input_features
"""Find the nearest value to a target value in a list of common values.

Parameters:
- target (int or float): The value to find the nearest value to.
- common (list[int or float]): A list of values to search for the nearest value from.

Returns:
- int or float: The nearest value to the target value.
"""
def nearest_val(target, common):
    return min(common, key=lambda x: abs(x - target))

# OneHotEncoderCustom Transformer
class OneHotEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, min_percentage=0.0):
        self.cols_to_encode = cols_to_encode
        self.min_percentage = min_percentage
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        self.train_value_counts_ = {}
        self.min_freq_category_ = {}
        self.min_freq_percentage_ = {}

        for col in self.cols_to_encode:
            value_counts = X[col].value_counts(normalize=True)
            self.train_value_counts_[col] = value_counts
            self.min_freq_category_[col] = value_counts.idxmin()
            self.min_freq_percentage_[col] = value_counts.min()
        
        return self
    
    def transform(self, X):
        X_encode = pd.DataFrame(index=X.index)
        for col in self.cols_to_encode:
            ohe_prefix = f"{col}_OHE"
            value_counts = self.train_value_counts_[col]
            min_freq_category = self.min_freq_category_[col]
            min_freq_category_name = f"{ohe_prefix}_{min_freq_category}"

            if self.min_freq_percentage_[col] < self.min_percentage:
                dummies = pd.get_dummies(X[col], prefix=ohe_prefix, prefix_sep='_')
                dummies.drop(columns=min_freq_category_name, errors='ignore', inplace=True)
            else:
                dummies = pd.get_dummies(X[col], prefix=ohe_prefix, prefix_sep='_')

            X_encode = pd.concat([X_encode, dummies], axis=1)
        
        non_encoded_cols = X.drop(columns=self.cols_to_encode)
        X_combined = pd.concat([non_encoded_cols, X_encode], axis=1)
        self.feature_names_out_ = X_combined.columns
        
        return X_combined

    def get_feature_names_out(self, input_features=None):
        encoded_cols = []
        for col in self.cols_to_encode:
            for value in self.train_value_counts_[col].index:
                encoded_cols.append(f"{col}_OHE_{value}")
        return encoded_cols
    
class ModelTrainer:
    def __init__(self, pipeline, Xtrain, Ytrain, Xtest, Ytest):
        self.pipeline = pipeline
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.label_encoder = LabelEncoder()
        self.best_models = {}
        self.best_model_accuracy = {}
        self.column_names = []
        
    def encode_labels(self):
        """
        Encode les étiquettes de classe en entiers.
        """
        self.Ytrain = self.label_encoder.fit_transform(self.Ytrain.ravel())
        self.Ytest = self.label_encoder.transform(self.Ytest.ravel())
    
    def train_models(self):
        """
        Entraîne les modèles de base sur les données d'entraînement et trouve les meilleurs hyperparamètres.
        """
        # Initialiser les classificateurs
        adaboost_clf = AdaBoostClassifier(random_state=42)
        extratrees_clf = ExtraTreesClassifier(random_state=42)
        gradientboost_clf = GradientBoostingClassifier(random_state=42)
        randomforest_clf = RandomForestClassifier(random_state=42)

        # Définir les grilles de recherche pour les hyperparamètres
        adaboost_params = {'n_estimators': [50, 100]}
        extratrees_params = {'n_estimators': [50, 100]}
        gradientboost_params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        randomforest_params = {'n_estimators': [50, 100]}
        
        # Initialiser les objets de recherche sur grille
        adaboost_grid = GridSearchCV(adaboost_clf, adaboost_params, cv=3, scoring='accuracy')
        extratrees_grid = GridSearchCV(extratrees_clf, extratrees_params, cv=5, scoring='accuracy')
        gradientboost_grid = GridSearchCV(gradientboost_clf, gradientboost_params, cv=5, scoring='accuracy')
        randomforest_grid = GridSearchCV(randomforest_clf, randomforest_params, cv=5, scoring='accuracy')

        # Effectuer la recherche sur grille pour chaque classificateur
        adaboost_grid.fit(self.Xtrain, self.Ytrain)
        extratrees_grid.fit(self.Xtrain, self.Ytrain)
        gradientboost_grid.fit(self.Xtrain, self.Ytrain)
        randomforest_grid.fit(self.Xtrain, self.Ytrain)

        # Stocker les meilleurs modèles
        self.best_models['adaboost'] = adaboost_grid.best_estimator_
        self.best_models['extratrees'] = extratrees_grid.best_estimator_
        self.best_models['gradientboost'] = gradientboost_grid.best_estimator_
        self.best_models['randomforest'] = randomforest_grid.best_estimator_
        
    def evaluate_models(self):
        """
        Évalue les modèles de base sur l'ensemble de validation.
        """
        for name, model in self.best_models.items():
            test_pred = model.predict(self.Xtest)
            test_accuracy = accuracy_score(self.Ytest, test_pred)
            self.best_model_accuracy[name] = test_accuracy
            print(f"{name.capitalize()} Accuracy on validation set: {test_accuracy}")
            
    def train_stacking_model(self):
        """
        Entraîne le modèle de stacking en utilisant les modèles de base comme estimateurs.
        """
        base_classifiers = [(name, model) for name, model in self.best_models.items()]
        self.stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=LogisticRegression())
        self.stacking_clf.fit(self.Xtrain, self.Ytrain)
        
    def evaluate_final_model(self):
        """
        Évaluation finale sur l'ensemble de test.
        """
        stacking_pred = self.stacking_clf.predict(self.Xtest)
        stacking_accuracy = accuracy_score(self.Ytest, stacking_pred)
        self.best_model_accuracy["stacking"] = stacking_accuracy
        print("Stacking Accuracy on test set:", stacking_accuracy)
        
    def save_best_model(self):
        """
        Enregistre le meilleur modèle en fonction du score dans self.best_model_accuracy.
        """
        best_model_name = max(self.best_model_accuracy, key=self.best_model_accuracy.get)
        best_model = self.best_models.get(best_model_name, self.stacking_clf)

        # joblib.dump(best_model, f"{best_model_name}_best_model.pkl")
        dh.save_pipeline(best_model)
        print(f"Best model {best_model_name} saved as {best_model_name}_best_model.pkl")

    def run(self):
        """
        Exécute le processus complet d'encodage, d'entraînement, d'évaluation et de test.
        """
        # Transformation des données d'entraînement et de test
        self.Xtrain_enc = self.pipeline.fit_transform(self.Xtrain, self.Ytrain)
        self.Xtest_enc = self.pipeline.transform(self.Xtest)
        
        # Récupérer les noms de colonnes après transformation
        self.column_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out(input_features=self.Xtrain.columns)
        self.column_names = [col.replace('remainder__', '').replace('one_hot_encoding__', '') for col in self.column_names]
        
        # Conversion des données transformées en DataFrame avec les noms de colonnes
        self.Xtrain = pd.DataFrame(self.Xtrain_enc, columns=self.column_names)
        self.Xtest = pd.DataFrame(self.Xtest_enc, columns=self.column_names)
        
        # Encodage des étiquettes
        self.encode_labels()
        
        # Entraînement et évaluation des modèles de base
        self.train_models()
        self.evaluate_models()
        
        # Entraînement et évaluation du modèle de stacking
        self.train_stacking_model()
        self.evaluate_final_model()
        
        # Enregistrer le meilleur modèle
        self.save_best_model()
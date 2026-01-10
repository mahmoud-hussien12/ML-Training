from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import os
import joblib
from src.utils import trim_values, to_float

def create_pipeline(numerical_features, categorical_features, model):
    numerical_transformer = Pipeline(steps=[
        ('fill_na', FunctionTransformer(trim_values)),
        ('to_float', FunctionTransformer(to_float)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return model_pipeline

def get_model(model_name: str, params: dict):
    if model_name == "logistic_regression":
        return LogisticRegression(**params)

def get_metrics(y_true, y_pred):
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    return metrics

def save_pipeline(pipeline, model_name, random_seed):
    os.makedirs("pipelines", exist_ok=True)
    joblib.dump(pipeline, f"pipelines/{model_name}_{random_seed}.joblib")
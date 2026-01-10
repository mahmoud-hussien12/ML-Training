from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import os
import joblib
from src.utils import trim_values, to_float
from lightgbm import LGBMClassifier

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
        ('classifier', model)
    ])
    return model_pipeline

def get_model(model_name: str, params: dict):
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    elif model_name == "lightgbm":
        return LGBMClassifier(**params)

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

def load_pipeline(model_name, random_seed):
    return joblib.load(f"pipelines/{model_name}_{random_seed}.joblib")

def get_feature_names(preprocessor):
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop":
            continue

        # 🔹 If this is a Pipeline, take its last step
        if hasattr(transformer, "steps"):
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(cols)
        else:
            names = cols

        feature_names.extend(names)

    return feature_names

import pandas as pd

def get_top_features(pipeline, top_k=10):
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    importances = None
    if isinstance(model, LGBMClassifier):
        importances = model.feature_importances_
    elif isinstance(model, LogisticRegression):
        importances = model.coef_[0]
    if importances is not None:
        feature_names = get_feature_names(preprocessor)
        assert len(importances) == len(feature_names)
        fi = (
            pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
                "absolute_importance": abs(importances)
            })
            .sort_values("absolute_importance", ascending=False)
            .head(top_k)
        )
        return fi
    return None

def evaluate_threshold(y_true, y_proba, threshold):
    y_true = np.asarray(y_true).ravel()
    y_proba_pos = y_proba[:, 1]
    y_pred = (y_proba_pos >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba_pos),
        "y_pred": y_pred
    }

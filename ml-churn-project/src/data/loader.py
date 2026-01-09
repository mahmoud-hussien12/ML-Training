# Data loader module
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_raw_data(path: str):
    """
    Loads raw churn CSV and returns a pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified file is not found: {path}")
    return pd.read_csv(path)

def split_data(
    df,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int
):
    """
    Splits the data into training, validation and test sets.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    val_size = val_size / (1-test_size)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val)

    return X_train, X_val, X_test, y_train, y_val, y_test
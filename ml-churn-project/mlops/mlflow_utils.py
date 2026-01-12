import mlflow
from pathlib import Path

def setup_mlflow(experiment_name: str):
    mlflow.set_tracking_uri("file://" + str(Path("mlruns").absolute()))
    mlflow.set_experiment(experiment_name)

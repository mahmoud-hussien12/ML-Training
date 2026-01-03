import yaml
import os

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified file is not found: {path}")
        
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

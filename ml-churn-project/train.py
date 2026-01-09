import argparse
import sys
import os

# Ensure the current directory is in the python path so we can import src
sys.path.append(os.getcwd())

from src import logger
from src import config
from src.data import loader
def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the config file")
    args = parser.parse_args()
    #load config
    logger.log_info(f"Loading config from {args.config}")
    cfg = config.load_config(args.config)
    logger.log_info(f"Config: {cfg['project']['name']}, seed: {cfg['project']['random_seed']}")
    #load raw data
    logger.log_info(f"Loading raw data from {cfg['data']['raw_path']}")
    df = loader.load_raw_data(cfg['data']['raw_path'])
    logger.log_info(f"Data loaded with shape: {df.shape}")
    #split data into train, val, test sets
    logger.log_info(f"Splitting data into train, val, test sets")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(df, cfg['data']['target_col'], cfg['training']['test_size'], cfg['training']['val_size'], cfg['project']['random_seed'])
    logger.log_info(f"Data split into train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    #log train, val, test counts
    counts, percentage = y_train.value_counts(), y_train.value_counts(normalize=True) * 100
    val_counts, val_percentage = y_val.value_counts(), y_val.value_counts(normalize=True) * 100
    test_counts, test_percentage = y_test.value_counts(), y_test.value_counts(normalize=True) * 100
    logger.log_info(f"Data train counts: {counts} with percentage: {percentage}")
    logger.log_info(f"Data val counts: {val_counts} with percentage: {val_percentage}")
    logger.log_info(f"Data test counts: {test_counts} with percentage: {test_percentage}")
    
if __name__ == "__main__":
    main()


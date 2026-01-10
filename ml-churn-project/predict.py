import argparse
import sys
import os

# Ensure the current directory is in the python path so we can import src
sys.path.append(os.getcwd())

from src import logger
from src import config
from src.data import loader
from src.models import pipeline as pipe
from src.features import build_features
from src.evaluation import cross_validation

def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the config file")
    args = parser.parse_args()
    #load config
    logger.log_info(f"Loading config from {args.config}")
    cfg = config.load_config(args.config)
    logger.log_info(f"Config: {cfg['project']['name']}, seed: {cfg['project']['random_seed']}")

    #load data
    logger.log_info("Loading data")
    test_df = loader.load_data(cfg['data']['test_path'])
    logger.log_info("Data loaded")

    #load model
    logger.log_info("Loading model")
    model = pipe.load_pipeline(cfg['training']['active_models'][0], cfg['project']['random_seed'])
    logger.log_info("Model loaded")

    #predict
    logger.log_info("Predicting")
    predictions = model.predict(test_df.drop(cfg['data']['target_col'], axis=1))
    logger.log_info("Predicted")

if __name__ == "__main__":
    main()
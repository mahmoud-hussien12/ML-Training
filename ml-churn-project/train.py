import argparse
import sys
import os

# Ensure the current directory is in the python path so we can import src
sys.path.append(os.getcwd())

from src import logger
from src import config

def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the config file")
    args = parser.parse_args()

    logger.log_info(f"Loading config from {args.config}")
    cfg = config.load_config(args.config)
    logger.log_info(f"Config: {cfg['project']['name']}, seed: {cfg['project']['random_seed']}")

if __name__ == "__main__":
    main()


import argparse
import sys
import os
import numpy as np
# Ensure the current directory is in the python path so we can import src
sys.path.append(os.getcwd())

from src import logger
from src import config
from src.data import loader
from src.models import pipeline as pipe
from src.features import build_features
from src.evaluation import cross_validation
from sklearn.metrics import confusion_matrix

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
    train_df = loader.load_data(cfg['data']['train_path'])
    test_df = loader.load_data(cfg['data']['test_path'])
    logger.log_info("Data loaded")

    #split data into X,y
    X_train = train_df.drop(cfg['data']['target_col'], axis=1)
    y_train = train_df[cfg['data']['target_col']]
    X_test = test_df.drop(cfg['data']['target_col'], axis=1)
    y_test = test_df[cfg['data']['target_col']]

    #load model
    logger.log_info("Loading model")
    model = pipe.load_pipeline('logistic_regression', cfg['project']['random_seed'])
    logger.log_info("Model loaded")

    #train
    model.fit(X_train, y_train)
    logger.log_info("Model trained")

    #predict
    logger.log_info("Predicting")
    y_proba = model.predict_proba(X_test)
    logger.log_info("Predicted")

    #save y_proba
    logger.log_info("Saving y_proba")
    np.save(cfg['data']['y_proba_path'], y_proba)
    logger.log_info("y_proba saved")
    #y_proba = np.load(cfg['data']['y_proba_path'])

    #threshold tuning
    logger.log_info("Thresholding")
    thresholds = cfg['training']['thresholds']
    y_pred = []
    max_recall = 0
    best_threshold = 0
    for threshold in thresholds:
        metrics = pipe.evaluate_threshold(y_test, y_proba, threshold)
        if metrics['recall'] > max_recall:
            max_recall = metrics['recall']
            y_pred = metrics['y_pred']
            best_threshold = threshold
        logger.log_info(f"Thresholded with precision: {metrics['precision']}, recall: {metrics['recall']}, roc_auc: {metrics['roc_auc']}, threshold: {threshold}")
    logger.log_info(f"selected threshold: {best_threshold} as it has the highest recall: {max_recall} which is decreaseing the customer churning by catching it then taking preventive actions")

    #confusion matrix
    logger.log_info("Confusion matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.log_info(f"Confusion matrix: {conf_matrix}")

    #save model artifact
    logger.log_info("Saving model artifact")
    pipe.save_pipeline(model, 'logistic_regression', cfg['project']['random_seed'])
    logger.log_info("Model artifact saved")

if __name__ == "__main__":
    main()
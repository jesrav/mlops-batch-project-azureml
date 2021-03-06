"""
Module to split modelling data into
- One data set for training and validation
- One hold out dataset for the final model performance evaluation
"""
import logging
from pathlib import Path

import hydra
import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd

from ..utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    logger.info("Fix seed.")
    seed = set_seed()
    mlflow.log_params({"seed": seed})

    logger.info('Load modelling data.')
    df = pd.read_parquet(
        Path(config["data"]["model_input"]["folder"]) 
        / config["data"]["model_input"]["file_name"]
    )

    logger.info('Split data in train/validate and test data.')
    train_validate_df, test_df = train_test_split(
        df,
        test_size=config["evaluation"]["test_set_ratio"],
    )

    logger.info('Save train/validate and test data.')
    train_validate_df.to_parquet(
        Path(config["data"]["train_validate_data"]["folder"]) 
        / config["data"]["train_validate_data"]["file_name"]
    )
    test_df.to_parquet(
        Path(config["data"]["test_data"]["folder"]) / config["data"]["test_data"]["file_name"]
    )
    

if __name__ == '__main__':
    main()
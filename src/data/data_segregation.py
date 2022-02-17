"""
Module to split modelling data into
- One data set for training and validation
- One hold out dataset for the final model performance evaluation
"""
import logging

import hydra
from sklearn.model_selection import train_test_split
import pandas as pd

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    df = pd.read_parquet(config["data"]["model_input"])

    logger.info('Split data in train/validate and test data.')
    train_validate_df, test_df = train_test_split(
        df,
        test_size=config["evaluation"]["test_set_ratio"],
    )

    logger.info('Save train/validate and test data.')
    train_validate_df.to_parquet(config["data"]["train_validate_data"])
    test_df.to_parquet(config["data"]["test_data"])
    

if __name__ == '__main__':
    main()
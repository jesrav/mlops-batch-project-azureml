"""
Module to add features.
"""
import logging

import hydra
import pandas as pd

logger = logging.getLogger(__name__)


def add_bedrooms_per_room(df: pd.DataFrame) -> pd.DataFrame:
    "Add average number of bedrooms per room."
    df = df.copy()
    df["avg_bedrooms_per_room"] = df.AveBedrms / df.AveRooms
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_bedrooms_per_room(df)
    return df


@hydra.main(config_path="../../conf", config_name="config")
def main(config):

    df = pd.read_parquet(config["data"]["clean_data"])

    logger.info('Add features.')
    df = add_features(df)

    logger.info('Save modelling input data.')
    df.to_parquet(config["data"]["model_input"])


if __name__ == "__main__":
    main()

"""
Module to do preprocessing.
"""
import logging
from pathlib import Path

import hydra
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    df = pd.read_parquet(
        Path(config["data"]["raw_data"]["folder"]) / config["data"]["raw_data"]["filename"]
    )

    logger.info('Preprocess raw artifacts.')
    df = preprocess(df)

    logger.info('Save preprocessed artifacts.')
    df.to_parquet(
        Path(config["data"]["clean_data"]["folder"]) / config["data"]["clean_data"]["filename"]
    )


if __name__ == "__main__":
    main()

"""
Module to do preprocessing of artifacts.
"""
import logging

import hydra
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
        df = pd.read_parquet(config["data"]["raw_data"])

        logger.info('Preprocess raw artifacts.')
        df = preprocess(df)

        logger.info('Log preprocessed artifacts.')
        df.to_parquet(config["data"]["clean_data"])


if __name__ == "__main__":
    main()

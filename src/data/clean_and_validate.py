"""
Module to do preprocessing.
"""
import logging
from pathlib import Path

import hydra
import pandas as pd
import pandera as pa

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df


def validate_model_input(df: pd.DataFrame) -> pd.DataFrame:
    schema_model_input = pa.DataFrameSchema({
        "MedInc": pa.Column(float, nullable=False, required=True),
        "HouseAge": pa.Column(float, nullable=False, required=True),
        "AveRooms": pa.Column(float, nullable=False, required=True),
        "Population": pa.Column(float, nullable=False, required=True),
        "AveOccup": pa.Column(float, nullable=False, required=True),
        "Latitude": pa.Column(float, nullable=False, required=True),
        "Longitude": pa.Column(float, nullable=False, required=True),
    })
    return schema_model_input.validate(df)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    df = pd.read_parquet(
        Path(config["data"]["raw_data"]["folder"]) / config["data"]["raw_data"]["file_name"]
    )

    logger.info('Preprocess raw artifacts.')
    df = preprocess(df)

    logger.info('Validate cleaned data.')
    df = validate_model_input(df)

    logger.info('Save preprocessed data.')
    df.to_parquet(
        Path(config["data"]["clean_data"]["folder"]) / config["data"]["clean_data"]["file_name"]
    )


if __name__ == "__main__":
    main()

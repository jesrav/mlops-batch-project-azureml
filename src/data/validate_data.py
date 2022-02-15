"""Module to validate model input data."""
import logging

import hydra
import pandas as pd
import pandera as pa

logger = logging.getLogger(__name__)


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
        logger.info('Read model input data.')
        df = pd.read_parquet(config["data"]["model_input"])

        logger.info('Validate model input.')
        df = validate_model_input(df)


if __name__ == "__main__":
    main()





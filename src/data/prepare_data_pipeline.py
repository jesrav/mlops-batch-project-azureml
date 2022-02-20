"""
Module to add features.
"""
import logging

import hydra

from .get_raw_data import get_raw_data
from .add_features import add_features
from .clean_and_validate import preprocess, validate_model_input

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    logger.info('Get raw data.')
    df = get_raw_data(
        sample_size=config["main"].get("inference_sample_size", None),
        med_inc_mean_drift_percentage=config["main"].get("med_inc_mean_drift_percentage", None)
    )

    logger.info('preprocess data.')
    df = preprocess(df)

    logger.info('Add features.')
    df = add_features(df)

    logger.info('Validate model input data.')
    df = validate_model_input(df)

    logger.info('Save model input data.')
    df.to_parquet(config["data"]["model_input"])


if __name__ == "__main__":
    main()

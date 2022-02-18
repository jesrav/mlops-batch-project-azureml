"""Module to do batch inference."""
import logging
import os 

import pandas as pd
import hydra
from azureml.core import Run

from ..utils import get_latest_model
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    workspace = Run.get_context().experiment.workspace    

    logger.info("Load data for batch predictions.")
    df = pd.read_parquet(config["data"]["model_input"])

    logger.info("Load model.")
    loaded_model = get_latest_model(
        workspace, 
        config["main"]["registered_model_name"], 
        tag_names=["prod"]
    )

    logger.info("Predict.")
    df['prediction'] = loaded_model.model.predict(df)
    df['model_id'] = loaded_model.model_meta_data.model_id

    logger.info("save predictions.")
    df.to_parquet(config['data']['predictions'])


if __name__ == '__main__':
    main()







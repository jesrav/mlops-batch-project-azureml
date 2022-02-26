"""Module to do batch inference."""
import logging
from pathlib import Path

import pandas as pd
import hydra
import mlflow
from azureml.core import Run

from ..utils import get_latest_model
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):  

    logger.info("Load data for batch predictions.")
    df = pd.read_parquet(
        Path(config["data"]["model_input"]["folder"]) / config["data"]["model_input"]["file_name"]
    )

    if config["main"]["run_locally"]:
        model_path = config["data"]["model"]["folder"]
        logger.info(f"Load model from local folder {model_path}.")
        loaded_model = mlflow.pyfunc.load_model(model_uri=str(model_path))
    else:
        logger.info("Load model from model registry.")
        workspace = Run.get_context().experiment.workspace  
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







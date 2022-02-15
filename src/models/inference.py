"""Module to do batch inference."""
import logging
import os 

import pandas as pd
import hydra
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

from .utils import get_latest_model
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(config):

    sp_auth = ServicePrincipalAuthentication(
        tenant_id=os.environ["TENANT_ID"],
        service_principal_id=os.environ["SERVICE_PRINCIPAL_ID"],
        service_principal_password=os.environ["SERVICE_PRINCIPAL_PASSWORD"],
    )
    workspace = Workspace.get(
        resource_group=os.environ["RESOURCE_GROUP"],
        name=os.environ["WORKSPACE_NAME"],
        auth=sp_auth,
        subscription_id=os.environ["SUBSCRIPTION_ID"],
    )

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







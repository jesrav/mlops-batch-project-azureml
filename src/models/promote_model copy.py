import os
import logging

import pandas as pd
import hydra
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import mlflow
from dotenv import load_dotenv, find_dotenv

from .utils import get_latest_model
from .evaluation import RegressionEvaluation

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


def get_model_performance(
    model: mlflow.pyfunc.PyFuncModel, 
    test_data: pd.DataFrame, 
    target_col: str,
    ) -> float:
    """Get model performance on hold out set."""
    predictions = model.predict(test_data)
    evaluation = RegressionEvaluation(
        y_true=test_data[target_col],
        y_pred=predictions
    )
    return evaluation.get_metrics()["mae"]


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

    logger.info("Load latest trained model.")
    latest_model = get_latest_model(workspace, config["main"]["registered_model_name"])

    logger.info("Load current production model if it exists")   
    try:
        current_prod_model = get_latest_model(workspace, config["main"]["registered_model_name"], tag_names=["prod"])
    except:
        current_prod_model = None 

    # If the current prod model is equal to the latest model, something went wrong.
    if current_prod_model and (latest_model.model_meta_data.model_id == current_prod_model.model_meta_data.model_id):
        raise ValueError("Latest model already in production")  

    logger.info("Load hold out data to test model performance on.")
    test_df = pd.read_parquet(config["data"]["test_data"])
    
    latest_model_performance = get_model_performance(
        model=latest_model.model,
        test_data=test_df,
        target_col=config["main"]["target_column"],
    )

    if latest_model_performance >= config["main"]["max_mae_to_promote"]:
        logger.warning(
            f"Latest model has an unacceptable performance of {latest_model_performance} mae, "
            f"which is above the threshold {config['main']['max_mae_to_promote']}. "  
            f"Not promoting model to production."
        )
    else:
        if not current_prod_model:    
            logger.info(
                f"Latest model has an acceptable performance of {latest_model_performance} mae on hold out set, "
                f"which is below threshold of {config['main']['max_mae_to_promote']}. "
                f"No current model in production. Promoting model to production"
            )
            latest_model.promote_to_prod()
        else:
            current_prod_model_performance = get_model_performance(
                model=current_prod_model.model,
                test_data=test_df,
                target_col=config["main"]["target_column"],
            )
            if current_prod_model_performance < latest_model_performance:
                logger.warning(
                    f"Latest model has a worse performance of on hold out set that current production model. " 
                    f"The latest model has {latest_model_performance} mae as compared to {current_prod_model_performance} for prod model. "
                    f"Not promoting model to production."
                )
            else:
                logger.info(
                    f"Latest model has a better performance of on hold out set that current production model. " 
                    f"The latest model has {latest_model_performance} mae as compared to {current_prod_model_performance} for prod model. "
                    f"Promoting model to production."
                )
                latest_model.promote_to_prod()
                current_prod_model.demote_from_prod()


if __name__ == "__main__":
    main()

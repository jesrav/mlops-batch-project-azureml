"""
Module for training and evaluating a model.

A model configuration that implements the interface found in
src.models.model_pipeliene_configs.BasePipelineConfig is passed supplied through the Hyrda configuration.
"""
import logging
from tempfile import TemporaryDirectory
from typing import Type
from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_val_predict
import hydra
import mlflow
from azureml.core import Run, Model

from ..utils import MLFlowModelWrapper, set_seed
from ..models.evaluation import RegressionEvaluation
from ..models import model_pipeliene_configs

logger = logging.getLogger(__name__)


def train_evaluate(
    df: pd.DataFrame,
    pipeline_class: Type[model_pipeliene_configs.BasePipelineConfig],
    config: dict,
):

    logger.info("Fix seed.")
    seed = set_seed()
    mlflow.log_params({"seed": seed})

    mlflow.log_params(config["model"]["params"])

    target_column = config["main"]["target_column"]

    logger.info("Initialize ml pipeline object.")
    pipeline = pipeline_class.get_pipeline(**(config["model"]["params"]))

    logger.info("predict on hold out data using cross validation.")
    predictions = cross_val_predict(
        estimator=pipeline,
        X=df,
        y=df[target_column],
        cv=config["evaluation"]["cross_validation_folds"],
        verbose=3,
    )

    model_evaluation = RegressionEvaluation(
        y_true=df[target_column],
        y_pred=predictions,
    )

    logger.info("train on model on all data")
    pipeline.fit(df, df[target_column])

    logger.info("Logging performance metrics.")
    mlflow.log_metrics(model_evaluation.get_metrics())

    logger.info("Logging model evaluation artifacts.")
    with TemporaryDirectory() as tmpdirname:
        model_evaluation.save_evaluation_artifacts(out_dir=tmpdirname)
        pipeline_class.save_fitted_pipeline_plots(pipeline, out_dir=tmpdirname)
        mlflow.log_artifact(tmpdirname, artifact_path="evaluation")

    if config["main"]["run_locally"]:
        model_path = config["data"]["model"]["folder"]
        logger.info(f"Saving model trained on all data to {model_path}.")
        mlflow.pyfunc.save_model(
            python_model=MLFlowModelWrapper(pipeline),
            path=model_path,
            conda_env=pipeline_class.get_conda_env(),
            code_path=["src"],
        )
    else:
        logger.info("Logging model trained on all data.")
        mlflow.pyfunc.log_model(
            python_model=MLFlowModelWrapper(pipeline),
            artifact_path="model",
            conda_env=pipeline_class.get_conda_env(),
            code_path=["src"],
            registered_model_name=config["main"]["registered_model_name"]
        )


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    logger.info("Load data for training model.")
    df = pd.read_parquet(
        Path(config["data"]["train_validate_data"]["folder"]) 
        / config["data"]["train_validate_data"]["file_name"]   
    )

    model_class = getattr(model_pipeliene_configs, config["model"]["ml_pipeline_config"])
    train_evaluate(
        df=df,
        pipeline_class=model_class,
        config=config,
    )


if __name__ == '__main__':
    main()
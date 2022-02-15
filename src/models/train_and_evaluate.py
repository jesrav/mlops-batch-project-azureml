import logging
from tempfile import TemporaryDirectory
from typing import Type

import pandas as pd
from sklearn.model_selection import cross_val_predict
import hydra
import mlflow

from ..models.utils import MLFlowModelWrapper
from ..models.evaluation import RegressionEvaluation
from ..models import models

logger = logging.getLogger(__name__)


def train_evaluate(
    pipeline_class: Type[models.BasePipelineConfig],
    config: dict,
):
    mlflow.log_params(config["model"]["params"])

    target_column = config["main"]["target_column"]

    logger.info("Load data for training model.")
    df = pd.read_parquet(config["data"]["train_validate_data"])

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

    logger.info("train on model on all artifacts")
    pipeline.fit(df, df[target_column])

    logger.info("Logging performance metrics.")
    mlflow.log_metrics(model_evaluation.get_metrics())

    logger.info("Logging model evaluation artifacts.")
    with TemporaryDirectory() as tmpdirname:
        model_evaluation.save_evaluation_artifacts(out_dir=tmpdirname)
        pipeline_class.save_fitted_pipeline_plots(pipeline, out_dir=tmpdirname)
        mlflow.log_artifact(tmpdirname, artifact_path="evaluation")

    logger.info("Logging model trained on all data.")
    mlflow.pyfunc.log_model(
        python_model=MLFlowModelWrapper(pipeline),
        artifact_path="model",
        conda_env=pipeline_class.get_conda_env(),
        code_path=["src"],
        registered_model_name=config["main"]["registered_model_name"] if not config["main"]["run_locally"] else None 
    )


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    model_class = getattr(models, config["model"]["ml_pipeline_config"])
    train_evaluate(
        pipeline_class=model_class,
        config=config,
    )


if __name__ == '__main__':
    main()
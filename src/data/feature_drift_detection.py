"""
Module for doing drift detection
"""
from tempfile import TemporaryDirectory
import logging
import os

import hydra
import pandas as pd
from azureml.core import Workspace, Model, Run, Experiment
from azureml.pipeline.core import PipelineRun
from azureml.core.authentication import ServicePrincipalAuthentication

from evidently.analyzers.data_drift_analyzer import DataDriftAnalyzer
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.model_profile import Profile
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)


def get_dataset_from_run(run: Run, dataset_name: str) -> pd.DataFrame:
    input_datasets = run.get_details()['inputDatasets']
    dataset = [
        dataset for dataset in input_datasets 
        if dataset["consumptionDetails"]["inputName"] == dataset_name
    ][0]["dataset"]

    with TemporaryDirectory() as tmpdirname:
        local_model_input_path = tmpdirname + "model_input.parquet"
        dataset.download(target_path=local_model_input_path, overwrite=False)
        df = pd.read_parquet(local_model_input_path)
    return df


def get_prod_model_training(workspace, experiment_name, model_name) -> pd.DataFrame:
    """Get training data used to train the current production model"""
    prod_aml_model = Model(workspace=workspace, name=model_name, tags=["prod"])
    experiment = Experiment(workspace, experiment_name)
    model_training_run = Run(experiment=experiment, run_id=prod_aml_model.run_id)
    return get_dataset_from_run(run=model_training_run, dataset_name="train_validate_data")
    

def get_latest_inference_data(workspace, experiment_name) -> pd.DataFrame:
    """Get training data used to train the current production model"""
    experiment = Experiment(workspace, experiment_name)
    inference_runs = experiment.get_runs()
    latest_inference_run = next(inference_runs)
 
    steps = latest_inference_run.get_steps()
    datasets = {}
    for step in steps:
        datasets = datasets | step.get_outputs()
    try: 
        inference_input_dataset = datasets["model_input"]
    except KeyError:
        raise KeyError("Model input dataset is not output from any of the pipeline steps.")

    dataset_reference = inference_input_dataset.get_port_data_reference()
    with TemporaryDirectory() as tmpdirname:
        dataset_reference.download(local_path=tmpdirname)
        file_path = (
            f"{tmpdirname}/{dataset_reference.path_on_datastore}/model_input.parquet"
        )
        return pd.read_parquet(file_path)


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

    training_data = get_prod_model_training(
        workspace=workspace,
        experiment_name=config["main"]["training_experiment_name"],
        model_name=config['main']['registered_model_name'],
    )

    # Get data supposed to represent a batch of recent data used for inference.
    # Most likely implemented as a rolling window. In this case we are just getting
    # data from the last batch inference.
    logger.info("Load data used for latest inference.")
    latest_inference_data = get_latest_inference_data(
        workspace=workspace, 
        experiment_name=config["main"]["inference_experiment_name"]
    )

    logger.info("Create and log data drift report.")
    data_drift_report = Dashboard(tabs=[DataDriftTab()])
    data_drift_report.calculate(
        reference_data=training_data,
        current_data=latest_inference_data
    )
    data_drift_report.save(config["data"]["feature_drift_report"])

    logger.info("Create and log data drift profile.")
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(
        reference_data=training_data,
        current_data=latest_inference_data
    )
    with TemporaryDirectory() as tmpdirname:
        with open(config["data"]["feature_drift_profile"], "w") as file:
            file.write(data_drift_profile.json())

    # Get number of drifted features from analyzer
    n_drifted_features = data_drift_profile.analyzers_results[DataDriftAnalyzer].metrics.n_drifted_features

    if n_drifted_features > 0:
        warning_text = (
            f"Feature drift detected for {n_drifted_features} features. "
            f"Check data drift report and profile in run."
        )
        logger.warning(warning_text)


if __name__ == '__main__':
    main()

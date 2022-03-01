"""
Module for doing drift detection
"""
from tempfile import TemporaryDirectory, tempdir
import logging

import hydra
import pandas as pd
import mlflow
from azureml.core import Model, Run, Experiment

from evidently.analyzers.data_drift_analyzer import DataDriftAnalyzer
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.model_profile import Profile

logger = logging.getLogger(__name__)


def get_dataset_from_run(run: Run, dataset_name: str) -> pd.DataFrame:
    input_datasets = run.get_details()['inputDatasets']
    dataset = [
        dataset for dataset in input_datasets 
        if dataset["consumptionDetails"]["inputName"] == dataset_name
    ][0]["dataset"]

    with TemporaryDirectory() as tmpdirname:
        local_model_input_path = f'{tmpdirname}model_input.parquet'
        dataset.download(target_path=local_model_input_path, overwrite=False)
        df = pd.read_parquet(local_model_input_path)
    return df


def get_prod_model_training_data(workspace, experiment_name, model_name) -> pd.DataFrame:
    """Get training data used to train the current production model"""
    prod_aml_model = Model(workspace=workspace, name=model_name, tags=["prod"])
    experiment = Experiment(workspace, experiment_name)
    model_training_run = Run(experiment=experiment, run_id=prod_aml_model.run_id)
    return get_dataset_from_run(run=model_training_run, dataset_name="train_validate_data")
    

def get_latest_inference_data(workspace, experiment_name) -> pd.DataFrame:
    """Get latest data use for batch inference."""
    experiment = Experiment(workspace, experiment_name)
    inference_runs = experiment.get_runs()
    latest_inference_run = next(inference_runs)
 
    pipeline_runs = list(latest_inference_run.get_children())
    batch_inference_run = [run for run in pipeline_runs if run.name == 'batch_inference'][0]
    return get_dataset_from_run(batch_inference_run, "model_input_inference")


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    workspace = Run.get_context().experiment.workspace

    training_data = get_prod_model_training_data(
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

    logger.info("Create and log data drift report and profile.")
    data_drift_report = Dashboard(tabs=[DataDriftTab()])
    data_drift_report.calculate(
        reference_data=training_data,
        current_data=latest_inference_data
    )
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(
        reference_data=training_data,
        current_data=latest_inference_data
    )
    with TemporaryDirectory() as tmpdirname:
        data_drift_report.save(f'{tmpdirname}/data_drift_report.html')
        with open(f'{tmpdirname}/data_drift_profile.json', "w") as file:
            file.write(data_drift_profile.json())
        mlflow.log_artifact(tmpdirname, artifact_path="drift-detection")

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

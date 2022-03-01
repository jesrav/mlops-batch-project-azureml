import os

from azureml.core import Workspace, Experiment
from azureml.pipeline.steps import CommandStep
from azureml.core.runconfig import RunConfiguration
from azureml.core import Environment 
from azureml.pipeline.core import Pipeline

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

################################################
# Setup
################################################
workspace = Workspace.get(
    resource_group=os.environ["RESOURCE_GROUP"],
    name=os.environ["WORKSPACE_NAME"],
    subscription_id=os.environ["SUBSCRIPTION_ID"],
)

datastore = workspace.get_default_datastore()

compute_target = workspace.compute_targets["cpu-cluster"]

aml_run_config = RunConfiguration()
aml_run_config.environment = Environment.get(
    workspace=workspace, name="mlops-example-env"
)

################################################
# Feature drift detection step
################################################
feature_drift_detection_step = CommandStep(
    name="feature_drift_detection",  
    command=(
        "python -m src.data.feature_drift_detection "
        "main=drift-detection-pipeline"
    ), 
    source_directory=".",
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

################################################
# Combine steps into feature drift detection pipeline
################################################
feature_drift_detection_pipeline = Pipeline(
    workspace=workspace, 
    steps=[feature_drift_detection_step],
)
# Submit feature drift detection job pipeline run
feature_drift_detection_pipeline_run = (
    Experiment(workspace, 'housing-model-drift-detection-pipeline')
    .submit(feature_drift_detection_pipeline)
)

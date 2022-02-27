import os

from azureml.core import Workspace, Experiment
from azureml.pipeline.steps import PythonScriptStep, CommandStep
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.data import OutputFileDatasetConfig
from azureml.core.runconfig import RunConfiguration
from azureml.core import Environment 
from azureml.pipeline.core import Pipeline, StepSequence

from dotenv import load_dotenv, find_dotenv

################################################
# Setup
################################################
load_dotenv(find_dotenv())

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

datastore = workspace.get_default_datastore()

compute_target = workspace.compute_targets["cpu-cluster"]

aml_run_config = RunConfiguration()
aml_run_config.environment = Environment.get(workspace=workspace, name="mlops-example-proj-env")

################################################
# Feature drift detection step
################################################
feature_drift_detection_step = CommandStep(
    name="feature_drift_detection",  
    command=("python -m src.data.feature_drift_detection"), 
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
feature_drift_detection_pipeline_run = Experiment(workspace, 'housing-model-drift-detection-pipeline').submit(feature_drift_detection_pipeline)

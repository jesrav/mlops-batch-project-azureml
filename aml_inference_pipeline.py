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
# Get and prepare data step
################################################
model_input_data = OutputFileDatasetConfig(name='model_input_inference')
model_input_data = model_input_data.register_on_complete(name='model_input_inference')

get_and_prepare_data_step = CommandStep(
    name="get_and_prepare_data",  
    command=(
        "python -m src.data.prepare_data_pipeline "
        "main=inference-pipeline "
        "data=inference-pipeline "
        f"data.model_input.folder={model_input_data.arg_val}"
    ), 
    source_directory=".",
    outputs=[model_input_data],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)


################################################
# Batch inference step
################################################
prediction_data = OutputFileDatasetConfig(name='prediction_data')
prediction_data = prediction_data.register_on_complete(name='prediction_data')
model_input_data_as_input = model_input_data.as_input(name="model_input_inference")

batch_inference_step = CommandStep(
    name="batch_inference", 
    command=(
        "python -m src.models.inference "
        "main=inference-pipeline "
        "data=inference-pipeline "
        f"data.model_input.folder={model_input_data_as_input.arg_val} "
        f"data.predictions.folder={prediction_data.arg_val}"
    ),
    source_directory=".",
    inputs=[model_input_data_as_input],
    outputs=[prediction_data],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

################################################
# Combine steps into batch inference pipeline
################################################
batch_inference_pipeline_steps = [
    get_and_prepare_data_step,
    batch_inference_step,
]
batch_inference_pipeline = Pipeline(
    workspace=workspace, 
    steps=batch_inference_pipeline_steps,
)
# Submit batch inference job pipeline run
batch_inference_pipeline_run = Experiment(workspace, 'housing-model-inference-pipeline').submit(batch_inference_pipeline)

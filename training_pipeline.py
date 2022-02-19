import os
from unicodedata import name

from azureml.core import Workspace, Experiment
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.data import OutputFileDatasetConfig
from azureml.core.runconfig import RunConfiguration
from azureml.core import Environment 
from azureml.pipeline.core import Pipeline

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

experiment=Experiment(workspace=workspace, name="housing-model-training-pipeline")

datastore = workspace.get_default_datastore()

compute_target = workspace.compute_targets["cpu-cluster"]

aml_run_config = RunConfiguration()
aml_run_config.environment = Environment.get(workspace=workspace, name="mlops-example-proj-env")
#aml_run_config.environment.name = "mlops-example-proj-env"
#aml_run_config.environment.version = "5"

################################################
# Get raw data step
################################################
raw_training_data = OutputFileDatasetConfig()
raw_training_data = raw_training_data.register_on_complete(name = 'raw_training_data')

get_raw_data_step = PythonScriptStep(
    name="get_raw_data",   
    script_name="src/data/get_raw_data.py",
    source_directory=".",
    arguments=[f"data.raw_data.folder={raw_training_data.arg_val}"],
    outputs=[raw_training_data],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)


################################################
# Preprocess data step
################################################
clean_training_data = OutputFileDatasetConfig()
clean_training_data = clean_training_data.register_on_complete(
    name = 'preprocesed_training_data'
)

preproces_training_data_step = PythonScriptStep(
    name="preprocess_data", 
    script_name="src/data/process_data.py",
    source_directory=".",
    arguments=[
        f"data.raw_data.folder={raw_training_data.as_input().arg_val}",
        f"data.clean_data.folder={clean_training_data.arg_val}",
    ],
    inputs=[raw_training_data.as_input()],
    outputs=[clean_training_data],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

################################################
# Add features step
################################################
model_input_data = OutputFileDatasetConfig()
model_input_data = model_input_data.register_on_complete(
    name = 'preprocesed_training_data'
)

add_features_step = PythonScriptStep(
    name="add_features",
    script_name="src/data/add_features.py",
    source_directory=".",
    arguments=[
        f"data.clean_data.folder={clean_training_data.as_input().arg_val}"
        f"data.model_input.folder={model_input_data.arg_val}"
    ],
    inputs=[clean_training_data.as_input()],
    outputs=[model_input_data],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)


steps = steps=[get_raw_data_step, preproces_training_data_step, add_features_step]
training_pipeline = Pipeline(
    workspace=workspace, 
    steps=steps,
)

training_pipeline_run = Experiment(workspace, 'test_pipeline_exp').submit(training_pipeline)

training_pipeline.wait_for_completion()
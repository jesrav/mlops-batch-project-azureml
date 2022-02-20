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
raw_training_data = OutputFileDatasetConfig(name="raw_data_training")
raw_training_data = raw_training_data.register_on_complete(name="raw_data_training")

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
# Clean and validate step
################################################
clean_training_data = OutputFileDatasetConfig(name='clean_data_training')
clean_training_data = clean_training_data.register_on_complete(name='clean_data_training')
raw_data_as_input = raw_training_data.as_input(name="raw_data_training")

clean_and_validate_step = PythonScriptStep(
    name="clean_and_validate", 
    script_name="src/data/clean_and_validate.py",
    source_directory=".",
    arguments=[
        f"data.raw_data.folder={raw_data_as_input.arg_val}",
        f"data.clean_data.folder={clean_training_data.arg_val}",
    ],
    inputs=[raw_data_as_input],
    outputs=[clean_training_data],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

################################################
# Add features step
################################################
model_input_data = OutputFileDatasetConfig(name='model_input_training')
model_input_data = model_input_data.register_on_complete(name='model_input_training')
clean_training_data_as_input = clean_training_data.as_input(name='clean_data_training')

add_features_step = PythonScriptStep(
    name="add_features",
    script_name="src/data/add_features.py",
    source_directory=".",
    arguments=[
        f"data.clean_data.folder={clean_training_data_as_input.arg_val}",
        f"data.model_input.folder={model_input_data.arg_val}",
    ],
    inputs=[clean_training_data_as_input],
    outputs=[model_input_data],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

################################################
# Data segregation step
################################################
train_validate_data = OutputFileDatasetConfig(name='train_validate_data')
train_validate_data = train_validate_data.register_on_complete(name='train_validate_data')
test_data = OutputFileDatasetConfig(name='test_data')
test_data = test_data.register_on_complete(name='test_data')
model_input_data_as_input = model_input_data.as_input(name='model_input_training')

data_segragation_step = CommandStep(
    name="data_segragation",
    command=(
        "python -m src.data.data_segregation "
        f"data.model_input.folder={model_input_data_as_input.arg_val} "
        f"data.train_validate_data.folder={train_validate_data.arg_val} "
        f"data.test_data.folder={test_data.arg_val} "
    ),
    source_directory=".",
    inputs=[model_input_data_as_input],
    outputs=[train_validate_data, test_data],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

################################################
# Train and evaluate step
################################################
train_validate_data_as_input = train_validate_data.as_input(name='train_validate_data')

train_and_evaluate_step = CommandStep(
    name="train_and_evaluate",
    command=(
        "python -m src.models.train_and_evaluate "
        f"data.train_validate_data.folder={train_validate_data_as_input.arg_val}"
    ),
    source_directory=".",
    inputs=[train_validate_data_as_input],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

################################################
# Test and promote model step
################################################
test_data_as_input = test_data.as_input(name="test_data")

test_and_promote_model_step = CommandStep(
    name="test_and_promote_model",
    command=(
        "python -m src.models.promote_model "
        f"data.test_data.folder={test_data_as_input.arg_val} "
    ),
    source_directory=".",
    inputs=[test_data_as_input],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)
test_and_promote_model_step.run_after(train_and_evaluate_step)

################################################
# Combine steps into training pipeline
################################################
training_steps = [
    get_raw_data_step,
    clean_and_validate_step, 
    add_features_step, 
    data_segragation_step,
    train_and_evaluate_step,
    test_and_promote_model_step,
]
training_pipeline = Pipeline(
    workspace=workspace, 
    steps=test_and_promote_model_step,
)
training_pipeline_run = Experiment(workspace, 'test_pipeline_exp').submit(training_pipeline)

#training_pipeline_run.wait_for_completion()

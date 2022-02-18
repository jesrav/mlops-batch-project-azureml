import os

from azureml.core import Workspace, Run, Experiment, Dataset
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import ServicePrincipalAuthentication
from dotenv import load_dotenv, find_dotenv

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

from azureml.core import Dataset
from azureml.data import OutputFileDatasetConfig
from azureml.core.runconfig import RunConfiguration
from azureml.core import Environment 

# Default datastore 
datastore = workspace.get_default_datastore()

compute_target = workspace.compute_targets["cpu-cluster"]

aml_run_config = RunConfiguration()
aml_run_config.environment = Environment.get(workspace=workspace, name="mlops-example-proj-env")

get_raw_data_source_dir = "./src/data"
entry_point = "get_raw_data.py"

output_data1 = OutputFileDatasetConfig(destination = (datastore, 'outputdataset/{run-id}'))
output_data_dataset = output_data1.register_on_complete(name = 'raw_training_data')


get_raw_data_step = PythonScriptStep(
    script_name=entry_point,
    source_directory=get_raw_data_source_dir,
    arguments=["--output", output_data1],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)
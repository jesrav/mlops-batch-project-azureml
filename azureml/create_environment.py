"""Create an AzureML environment from the conda spec."""
import os

from azureml.core import Environment, Workspace
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

workspace = Workspace.get(
    resource_group=os.environ["RESOURCE_GROUP"],
    name=os.environ["WORKSPACE_NAME"],
    subscription_id=os.environ["SUBSCRIPTION_ID"],
)
aml_env = Environment.from_conda_specification(
    name="mlops-example-env",
    file_path="conda.yml"
)
aml_env.register(workspace=workspace)
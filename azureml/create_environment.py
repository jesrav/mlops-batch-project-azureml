"""Create an AzureML environment from the conda spec."""
import os

from azureml.core import Environment, Workspace
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
aml_env = Environment.from_conda_specification(
    name="mlops-example-env",
    file_path="conda.yml"
)
aml_env.register(workspace=workspace)
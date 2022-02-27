"""utils for using MLFlow and Azure ML."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import tempfile

import mlflow
import numpy as np
from azureml.core import Model, Workspace
from azureml.core.authentication import AzureCliAuthentication
import mlflow.pyfunc


class MLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for creating a MLFlow pyfunc from a fitted model,
     with a predict method
     """
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


@dataclass
class ModelMetaData:
    """Class for holding metadata on registered models."""
    model_id: str
    run_id: str


@dataclass
class LoadedMLFlowModel:
    """Class for holding both a mlflow pyfunc model and meta data
    on the registered model.
    """
    model: mlflow.pyfunc.PyFuncModel
    model_meta_data: ModelMetaData
    aml_model: Union[Model, None]

    @classmethod
    def from_aml_model(cls, aml_model: Model) -> 'LoadedMLFlowModel':
        """Get a `LoadedModel`from a Azure ML Model"""
        model_meta_data = ModelMetaData(model_id=aml_model.id, run_id=aml_model.run_id)
        temp_dir = tempfile.mkdtemp()
        aml_model.download(temp_dir, exist_ok=True)
        model = mlflow.pyfunc.load_model("file:" + str(temp_dir / Path("model")))
        return LoadedMLFlowModel(model=model, model_meta_data=model_meta_data, aml_model=aml_model)

    @classmethod
    def from_local_path(cls, model_path: str) -> 'LoadedMLFlowModel':
        """Get a `LoadedModel`from a local path.
        
        Model version, run id and aml model object are not available in this case.
        """
        model = mlflow.pyfunc.load_model(model_uri=model_path)
        return LoadedMLFlowModel(
            model=model, 
            model_meta_data=ModelMetaData(model_id="none", run_id="none"), 
            aml_model=None
        )

    def promote_to_prod(self):
        """Promote model to production"""
        if not self.aml_model:
            raise ValueError("Model not registered in AML. Can not promote.")
        self.aml_model.add_tags({"prod": True})
        self.aml_model.update_tags_properties()

    def demote_from_prod(self):
        """Demote model from production"""
        if not self.aml_model:
            raise ValueError("Model not registered in AML. Can not demote.")
        self.aml_model.remove_tags(["prod"])
        self.aml_model.update_tags_properties()


def get_model_version(
    workspace: Workspace,
    model_name: str,
    model_version: int = None,
) -> LoadedMLFlowModel:
    """Get specific model version and dictionary with meta data about the model.
    If no model version is specified, the newest model is returned.
    Parameters
    ----------
    workspace:
        Azure ML workspace
    model_name:
        Name of registered model
    model_version:
        Version of registered model
    Returns
    -------
    LoadedMLFlowModel:
        Object with model and dictionary with model meta data.
    """
    aml_model = Model(workspace=workspace, name=model_name, version=model_version)
    return LoadedMLFlowModel.from_aml_model(aml_model)


def get_latest_model(
    workspace: Workspace,
    model_name: str,
    tag_names: Union[List[str], None] = None,
) -> LoadedMLFlowModel:
    """
    Get latest model with a specific tag and a dictionary with
    meta data about the model.
    Parameters
    ----------
    workspace:
        Azure ML workspace
    model_name:
        Name of registered model
    tag_names:
        Tags required for model. If no tags are passed, we do not filter on tags.
    Returns
    -------
    LoadedMLFlowModel:
        Object with model and dictionary with model meta data.
    """
    aml_model = Model(workspace=workspace, name=model_name, tags=tag_names)
    return LoadedMLFlowModel.from_aml_model(aml_model)


def set_seed(seed=33):
    np.random.seed(seed)
    return seed


def set_mlflow_uri_cli_auth(subscription_id, resource_group, name):
    cli_auth = AzureCliAuthentication()
    workspace = Workspace.get(
        resource_group=resource_group,
        name=name,
        auth=cli_auth,
        subscription_id=subscription_id,
    )
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
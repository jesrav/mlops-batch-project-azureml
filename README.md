# MLOps batch example project, using Azure ML

Toy project, that implements passive retraining and drift detection for a batch prediction regression use case.

The project deploys pipelines for training, inference and drift detection as Azure ML pipeline jobs.



## Tools used

- [Hydra](https://hydra.cc) for configuration 
- [MLFlow](https://mlflow.org) and AzureMl for experiment tracking and model packaging/versioning
- [AzureMl](https://docs.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learn) for data versioning through AzureML Pipelines
- [Evidently AI](https://evidentlyai.com) for drift detection



## Running locally / on Azure

Azure ml has some great functionality for tracking experiments and versioning data and models.
Unfortunately you can not run Azure ml pipelines locally. To be able to iterate quickly, all the steps in the ml training and inference pipelines are python command line scripts that can be orchestrated locally using make. When running locally, we write the mlflow runs to the standard local folder mlruns.
To run the pipelines in azure, we create and submit azureml pipelines, combining the script steps using the azureml SDK.



# Get started

## Requirements

- Azure account
- AzureML workspace
- The Azure CLI. See [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) for installation.
- The Azure ml CLI extension. See [here](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli) for installation.

## Install dependencies

```bash
conda env create --file conda.yml 
```

## Setup environment variables

Copy the `.env_example` template file to `.env` and fill out the environment variables.



## Run pipelines locally

When running the training and inference pipeline locally, output data will be stored in the`data` folder and logging from the mlflow runs will be stored in the `mlruns` folder.

### Training pipeline 

```bash
make local_train_pipeline
```

### Inference pipeline

```bash
make local_inference_pipeline
```



## Run as AzureML pipelines

When submitting the pipeline runs you will be interactively prompted to log in to your azure account tho authenticate.

In a production setting you should use a [service principal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication).


### Set azure ml defaults
```bash
make set_aml_deafaults
```

### Create AzureML compute cluster
```bash
make create_aml_compute
```

### Create AzureML environment
```bash
make create_aml_env
```

### Run training pipeline
```bash
make aml_train_pipeline
```
This will run a training pipeline that will train a model, test it and potentially promote it to production status (by tagging the model arrtifact with a `prod` tag.

### Run inference pipeline
```bash
make run_aml_inference_job
```
This will run an inference pipeline that will use the `prod` model to make predictions on new data (just a sample from the Boston housing data).
A very simplistic drift can be configures in the `main` Hydra configuration.

### Run drift detection on newest predictions
```bash
make run_aml_drift_detection_job
```
This will run drift detection, that compares the data used to make the latest predictions with the data used to train the latest `prod` model.



# See runs

To see your runs go to https://ml.azure.com/ and log into your workspace.

### Note on AzureML CLI v2

AzureML has a new CLI in preview. It does not seem to implement all the functionality available using the python SDK, but is really interesting. The pipelines can be run using this cli using the yaml files in `azureml_cli_v2`. See how to install the azureml cli v2 [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).




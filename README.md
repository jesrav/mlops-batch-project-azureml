# MLOps batch example project, using Azure ML CLI V2

Toy project, that implements passive retraining for a batch prediction regression use case.

The project is not concerned with deployment or the development of the ML models. 
It uses the Boston housing data and can be run locally.


## Tools used
- [Hydra](https://hydra.cc) for configuration 
- [MLFlow](https://mlflow.org) and AzureMl for experiment tracking and model packaging/versioning
- [AzureMl](https://docs.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learn) for data versioning through AzureML Pipelines
- [Evidently AI](https://evidentlyai.com) for drift detection


# Get started

## Requirements
- Azure account
- AzureML workspace
- the Azure CLI. See [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) for installation.
- The Azure ML CLI version 2 extension. See [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli) on how to install.


## Run project

## Set environment variable 
Copy the `.env_example` to `.env` and fill out the environment variables.

### Set azure cli defaults
```bash
make set_az_deafaults
```

### Create AzureML compute cluster


### Create AzureML environment
```bash
make create_aml_env
```

### Run training pipeline
```bash
make run_aml_train_job
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




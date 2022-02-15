include .env
export


###############################################################
# Train pipeline
###############################################################
train_pipeline: get_raw_data_train preprocess_data_train add_features_train validate_model_input_train
train_pipeline: data_segregation train_random_forest test_and_promote_model

get_raw_data_train:
	python -m src.data.get_raw_data

preprocess_data_train:
	python -m src.data.process_data
add_features_train:
	python -m src.data.add_features

validate_model_input_train:
	python -m src.data.validate_data

data_segregation:
	python -m src.data.data_segregation

train_ridge:
	python -m src.models.train_and_evaluate model=ridge main.run_locally=true

train_random_forest:
	python -m src.models.train_and_evaluate model=random_forest main.run_locally=true

test_and_promote_model:
	python -m src.models.promote_model

###############################################################
# Inference pipeline
###############################################################
inference_pipeline: get_raw_data_inference preprocess_data_inference add_features_inference
inference_pipeline: validate_model_input_inference batch_inference

get_raw_data_inference:
	python -m src.data.get_raw_data main=inference-pipeline data=inference-pipeline

preprocess_data_inference:
	python -m src.data.process_data main=inference-pipeline data=inference-pipeline

add_features_inference:
	python -m src.data.add_features main=inference-pipeline data=inference-pipeline

validate_model_input_inference:
	python -m src.data.validate_data main=inference-pipeline data=inference-pipeline

batch_inference:
	python -m src.models.inference.py main=inference-pipeline data=inference-pipeline


###############################################################
# Drift detection pipeline
###############################################################
make drift_detection:
	python -m src.data.feature_drift_detection main=drift-detection-pipeline data=drift-detection-pipeline


###############################################################
# Utils
###############################################################
set_az_deafaults:
	sudo az configure --defaults group="mlops-example" workspace="mlops-example" location="westeurope"

create_aml_env:
	sudo az ml environment create --file aml_environment.yml

run_aml_train_job:
	sudo az ml job create -f jobs/training-job.yml --set \
	jobs.test_and_promote_model.environment_variables.SUBSCRIPTION_ID=$(SUBSCRIPTION_ID) \
	jobs.test_and_promote_model.environment_variables.TENANT_ID=$(TENANT_ID) \
	jobs.test_and_promote_model.environment_variables.RESOURCE_GROUP=$(RESOURCE_GROUP) \
	jobs.test_and_promote_model.environment_variables.WORKSPACE_NAME=$(WORKSPACE_NAME) \
	jobs.test_and_promote_model.environment_variables.SERVICE_PRINCIPAL_ID=$(SERVICE_PRINCIPAL_ID) \
	jobs.test_and_promote_model.environment_variables.SERVICE_PRINCIPAL_PASSWORD=$(SERVICE_PRINCIPAL_PASSWORD) 

run_aml_inference_job:
	sudo az ml job create -f jobs/inference-job.yml --set \
	jobs.inference.environment_variables.SUBSCRIPTION_ID=$(SUBSCRIPTION_ID) \
	jobs.inference.environment_variables.TENANT_ID=$(TENANT_ID) \
	jobs.inference.environment_variables.RESOURCE_GROUP=$(RESOURCE_GROUP) \
	jobs.inference.environment_variables.WORKSPACE_NAME=$(WORKSPACE_NAME) \
	jobs.inference.environment_variables.SERVICE_PRINCIPAL_ID=$(SERVICE_PRINCIPAL_ID) \
	jobs.inference.environment_variables.SERVICE_PRINCIPAL_PASSWORD=$(SERVICE_PRINCIPAL_PASSWORD)

run_aml_drift_detection_job:
	sudo az ml job create -f jobs/drift-detection-job.yml --set \
	jobs.drift_detection.environment_variables.SUBSCRIPTION_ID=$(SUBSCRIPTION_ID) \
	jobs.drift_detection.environment_variables.TENANT_ID=$(TENANT_ID) \
	jobs.drift_detection.environment_variables.RESOURCE_GROUP=$(RESOURCE_GROUP) \
	jobs.drift_detection.environment_variables.WORKSPACE_NAME=$(WORKSPACE_NAME) \
	jobs.drift_detection.environment_variables.SERVICE_PRINCIPAL_ID=$(SERVICE_PRINCIPAL_ID) \
	jobs.drift_detection.environment_variables.SERVICE_PRINCIPAL_PASSWORD=$(SERVICE_PRINCIPAL_PASSWORD)
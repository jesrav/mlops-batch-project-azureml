include .env
export

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
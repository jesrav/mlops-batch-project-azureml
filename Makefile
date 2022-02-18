include .env
export

set_az_deafaults:
	sudo az configure --defaults group="mlops-example" workspace="mlops-example" location="westeurope"

create_aml_compute:
	sudo az ml compute create --file azureml/aml_compute.yml
	
create_aml_env:
	sudo az ml environment create --file azureml/aml_environment.yml

run_aml_train_job:
	sudo az ml job create -f azureml/training-job.yml

run_aml_inference_job:
	sudo az ml job create -f azureml/inference-job.yml

run_aml_drift_detection_job:
	sudo az ml job create -f azureml/drift-detection-job.yml
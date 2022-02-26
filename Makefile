include .env
export

###############################################################
# Train pipeline
###############################################################
train_pipeline: get_raw_data_train clean_and_validate_train add_features_train
train_pipeline: data_segregation train_random_forest

get_raw_data_train:
	python -m src.data.get_raw_data main.run_locally=true

clean_and_validate_train:
	python -m src.data.clean_and_validate main.run_locally=true

add_features_train:
	python -m src.data.add_features main.run_locally=true

data_segregation:
	python -m src.data.data_segregation main.run_locally=true

train_ridge:
	python -m src.models.train_and_evaluate model=ridge main.run_locally=true

train_random_forest:
	python -m src.models.train_and_evaluate model=random_forest main.run_locally=true


###############################################################
# Inference pipeline
###############################################################
inference_pipeline: prepare_data_pipeline batch_inference

prepare_data_pipeline:
	python -m src.data.prepare_data_pipeline main=inference-pipeline data=inference-pipeline main.run_locally=true

batch_inference:
	python -m src.models.inference main=inference-pipeline data=inference-pipeline main.run_locally=true


###############################################################
# Drift detection pipeline
###############################################################
make drift_detection:
	python -m src.data.feature_drift_detection main=drift-detection-pipeline data=drift-detection-pipeline


###############################################################
# Azureml
###############################################################
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
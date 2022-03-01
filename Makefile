include .env
export

###############################################################
# Local train pipeline
###############################################################
local_train_pipeline: get_raw_data_train clean_and_validate_train add_features_train
local_train_pipeline: data_segregation train_random_forest

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
# Local inference pipeline
###############################################################
local_inference_pipeline: prepare_data_pipeline batch_inference

prepare_data_pipeline:
	python -m src.data.prepare_data_pipeline main=inference-pipeline data=inference-pipeline main.run_locally=true

batch_inference:
	python -m src.models.inference main=inference-pipeline data=inference-pipeline main.run_locally=true


###############################################################
# Azureml setup
###############################################################
set_aml_deafaults:
	az ml folder attach -w "mlops-example" -g "mlops-example"

create_aml_compute:
	az ml computetarget create amlcompute \
	--name cpu-cluster \
	--vm-size STANDARD_DS11_V2 \
    --min-nodes 0 \
    --max-nodes 1 \
    --idle-seconds-before-scaledown 120

create_aml_env:
	python azureml/create_environment.py

###############################################################
# Azureml pipelines
###############################################################
aml_train_pipeline:
	python azureml/training_pipeline.py

aml_inference_pipeline:
	python azureml/inference_pipeline.py

aml_drift_detection_pipeline:
	python azureml/drift_detection_pipeline.py

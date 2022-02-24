"""
Script for promoting latest trained model to production if the performance on a hold out set:
- is better than a fixed threshold.
- is better than the current production model.
"""
import logging
from pathlib import Path

import hydra
from azureml.core import Run
import pandas as pd

from .evaluation import RegressionEvaluation
from ..utils import get_latest_model


logger = logging.getLogger(__name__)


def log_promotion_status(model_id: str, additional_info: str, model_to_be_promoted: bool) -> None:
    if model_to_be_promoted:
        message = f"Trained model version {model_id} promoted."
        logger.info(
            f"{message} "
            f"{additional_info}"
        )
    else:
        message = f"Trained model with version {model_id} not promoted."
        logger.warning(
            f"{message} "
            f"{additional_info}"
        )


class SingleModelTest:
    """Tests of a single model"""
    def __init__(self, model, test_data, target_col, max_mae):
        self.model = model
        self.test_data = test_data
        self.target_col = target_col
        self.max_mae = max_mae
        self.predictions = model.predict(test_data)
        self.model_mae = self._calc_model_mae(self.predictions, test_data, target_col)

    @staticmethod
    def _calc_model_mae(predictions, test_data, target_col):
        evaluation = RegressionEvaluation(
            y_true=test_data[target_col],
            y_pred=predictions
        )
        return evaluation.get_metrics()["mae"]

    def _model_has_ok_mae(self) -> bool:
        return self.model_mae < self.max_mae

    def _predictions_all_positive(self) -> bool:
        return self.predictions.min() > 0

    @property
    def model_passes_tests(self) -> bool:
        return all([
            self._model_has_ok_mae(), self._predictions_all_positive()
        ])

    @property
    def message(self):
        if self._model_has_ok_mae:
            mae_message = (
                f"The model has MAE of {self.model_mae}, which is under the max threshold of {self.max_mae}"
            )
        else:
            mae_message = (
                f"The model has MAE of {self.model_mae}, which is not below the max threshold of {self.max_mae}"
            )
        edge_case_message = (
            "The model passes all edge cases" if self._predictions_all_positive else "The model does not pass all edge cases"
        )
        return f"{mae_message}. {edge_case_message}."


class ChallengerModelTest:
    """Tests that compare a new challenger model to another model."""
    def __init__(self, model_challenger, model_current, test_data, target_col):
        self.model_challenger = model_challenger
        self.model_current = model_current
        self.test_data = test_data
        self.target_col = target_col
        self.model_challenger_mae = self._calc_model_mae(model_challenger, test_data, target_col)
        self.model_current_mae = self._calc_model_mae(model_current, test_data, target_col)

    @staticmethod
    def _calc_model_mae(model, test_data, target_col):
        predictions = model.predict(test_data)
        evaluation = RegressionEvaluation(
            y_true=test_data[target_col],
            y_pred=predictions
        )
        return evaluation.get_metrics()["mae"]

    @property
    def challenger_model_is_better(self) -> bool:
        return self.model_challenger_mae < self.model_current_mae

    @property
    def message(self):
        if self.challenger_model_is_better:
            mae_message = (
                f"Challenger model has MAE of {self.model_challenger_mae}, "
                f"which is better than the current models performance of {self.model_current_mae}"
            )
        else:
            mae_message = (
                f"Challenger model has MAE of {self.model_challenger_mae}, "
                f"which is worse than the current models performance of {self.model_current_mae}"
            )

        return f"{mae_message}."


@hydra.main(config_path="../../conf", config_name="config")
def main(config):

    workspace = Run.get_context().experiment.workspace
    
    logger.info("Load hold out test data.")
    test_data = pd.read_parquet(
        Path(config["data"]["test_data"]["folder"]) 
        / config["data"]["test_data"]["file_name"]  
    )

    logger.info("Loading latest trained model.")
    loaded_model_challenger = get_latest_model(workspace, config["main"]["registered_model_name"])

    #TODO: raise error if tag is prod

    logger.info("Loading current prod model if it exists.")
    try:
        loaded_model_current = get_latest_model(workspace, config["main"]["registered_model_name"], tag_names=["prod"])
    except:
        loaded_model_current = None 

    logger.info("Running single model tests.")
    single_model_test = SingleModelTest(
        model=loaded_model_challenger.model,
        test_data=test_data,
        target_col=config["main"]["target_column"],
        max_mae=config["main"]["max_mae_to_promote"]
    )

    if not loaded_model_current:
        model_to_be_promoted = single_model_test.model_passes_tests
        if model_to_be_promoted:
            loaded_model_challenger.promote_to_prod()

        log_promotion_status(
            model_id=loaded_model_challenger.model_meta_data.version,
            additional_info=(
                f"{single_model_test.message}"
            ),
            model_to_be_promoted=model_to_be_promoted
        )
    else:
        logger.info("Running model challenger comparison tests.")
        challenger_model_test = ChallengerModelTest(
            model_challenger=loaded_model_challenger.model,
            model_current=loaded_model_current.model,
            test_data=test_data,
            target_col=config["main"]["target_column"],
        )

        model_to_be_promoted = (
                single_model_test.model_passes_tests
                and challenger_model_test.challenger_model_is_better
        )
        if model_to_be_promoted:
            loaded_model_challenger.promote_to_prod()
            loaded_model_current.demote_from_prod()

        log_promotion_status(
            model_id=loaded_model_challenger.model_meta_data.model_id,
            additional_info=(
                f"{single_model_test.message} "
                f"{challenger_model_test.message}"
            ),
            model_to_be_promoted=model_to_be_promoted
        )


if __name__ == '__main__':
    main()
"""
Module that holds the ml model configs.
Any model config that needs to work with the `src/modelling/train_evaluate.py` module,
must conform to this interface specified in the meta class BasePipelineConfig.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pandera as pa
import seaborn as sns
from matplotlib import pyplot as plt

from .custom_transfomer_classes import ColumnSelector


class BasePipelineConfig(ABC):
    """Base class for ml model configs."""

    @staticmethod
    @abstractmethod
    def get_pipeline(**params):
        """Returns SKLearn compatible ml pipeline
        input:
            params: Parameters for sklearn compatible pipeline.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_conda_env() -> dict:
        """Get conda environment spec"""
        pass

    @staticmethod
    @abstractmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        """Saves any plots that are relevant for the fitted pipeline."""
        pass


class RidgePipelineConfig(BasePipelineConfig):
    """Model config for ML pipeline using a logistic regression model."""

    @staticmethod
    def get_pipeline(**params):
        """Get logistic regression pipeline
        The pipeline works on a dataframe and selects the features.
        input:
            params: Parameters for the sklearn compatible pipeline.
        """
        features = [
            'MedInc',
            'HouseAge',
            'AveRooms',
            'AveBedrms',
            'Population',
            'AveOccup',
            'Latitude',
            'Longitude',
            'avg_bedrooms_per_room',
        ]

        # Define pipeline
        vanilla_pipeline = Pipeline([
            ("column_selector", ColumnSelector(features)),
            ("regressor", Ridge()
             )
        ])
        return deepcopy(vanilla_pipeline).set_params(**params)

    @staticmethod
    def get_conda_env() -> dict:
        """Get conda environment spec"""
        return {
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9",
                "scikit-learn==1.0.2",
                "pip",
                {
                    "pip": [
                        "mlflow==1.23.1",
                    ],
                },
            ],
            "name": "ridge-model-env",
        }
    
    @staticmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        """Logreg pipeline does not have any plots for the fitted model."""
        pass


class RandomForestPipelineConfig(BasePipelineConfig):
    """Model config for ML pipeline using a random forest model."""

    @staticmethod
    def get_pipeline(**params):
        """Get random forest pipeline
        The pipeline works on a dataframe and selects the features.
        input:
            params: Parameters for the sklearn compatible pipeline.
        """
        features = [
            'MedInc',
            'HouseAge',
            'AveRooms',
            'AveBedrms',
            'Population',
            'AveOccup',
            'Latitude',
            'Longitude',
            'avg_bedrooms_per_room',
        ]

        # Define pipeline
        vanilla_pipeline = Pipeline([
            ("column_selector", ColumnSelector(features)),
            ("regressor", RandomForestRegressor()
             )
        ])
        return deepcopy(vanilla_pipeline).set_params(**params)

    @staticmethod
    def get_conda_env() -> dict:
        """Get conda environment spec"""
        return {
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9",
                "scikit-learn1.0.2",
                "pip",
                {
                    "pip": [
                        "mlflow==1.23.1",
                    ],
                },
            ],
            "name": "random-forest-model-env",
        }
    
    @staticmethod
    def save_fitted_pipeline_plots(pipeline, out_dir: str):
        """Save plot of feature importances for random forest model."""
        rf_features = pipeline["regressor"].feature_names_in_
        rf_feature_importances = pipeline["regressor"].feature_importances_
        feature_importance_df = pd.DataFrame(
            zip(
                rf_features,
                rf_feature_importances,
            ),
            columns=["feature", "importance"],
        ).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(20, 20))
        ax = sns.barplot(x="feature", y="importance", data=feature_importance_df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_xticklabels(), fontsize=24)
        plt.setp(ax.get_yticklabels(), fontsize=24)
        plt.xlabel('feature', fontsize=24)
        plt.ylabel('importance', fontsize=24)
        fig = ax.get_figure()
        fig.subplots_adjust(bottom=0.3)
        fig.savefig(Path(out_dir) / Path("random_forest_feature_importances.png"))

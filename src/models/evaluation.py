"""Module that contains functionality to evaluate model performance.
The main function is evaluate, which returns metrics and plots about out of sample predictions.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


class RegressionEvaluation:
    """Class to do evaluation on the performance of a regression model."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Construct the Evaluation object
        :y_true: y_true (array-like, shape (n_samples)) – Ground truth (correct) target values.
        :y_pred (array-like, shape (n_samples)) – Predictions from the regressor
        :return: None
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred must be the same.")
        self.y_true=y_true
        self.y_pred = y_pred

    def get_metrics(self) -> dict:
        return {
            "mse": mean_squared_error(self.y_true, self.y_pred),
            "mape": mean_absolute_percentage_error(self.y_true, self.y_pred),
            "mae": mean_absolute_error(self.y_true, self.y_pred),
        }

    def plot_actual_vs_predictions(self, outpath: Path, log_scale=False) -> None:
        """Plot actual values vs. predictions
        The plot is saved to outpath
        :outpath: Outpath for plot
        :log_scale: Whether to use a log scale for the axis
        :return: None
        """
        fig, ax = plt.subplots()
        plt.scatter(self.y_true, self.y_pred, c='crimson')
        if log_scale:
            plt.yscale('log')
            plt.xscale('log')
        p1 = max(max(self.y_pred), max(self.y_true))
        p2 = min(min(self.y_pred), min(self.y_true))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')
        plt.savefig(str(outpath))
        plt.close()

    def save_evaluation_artifacts(self, out_dir: Path) -> None:
        """Save all evaluation artifacts to a folder"""
        self.plot_actual_vs_predictions(out_dir / Path("actual_vs_predictions_plot.png"))
        with open(out_dir / Path("metrics.json"), "w") as f:
            json.dump(self.get_metrics(), f)

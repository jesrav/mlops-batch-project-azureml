"""Module that contains functionality to evaluate model performance.
The main function is evaluate, which returns metrics and plots about out of sample predictions.
"""
import json
from pathlib import Path

import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
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



class ClassificationEvaluation:
    """Class to do evaluation on the performance of a classification model."""

    def __init__(
            self, y_true: np.ndarray, y_proba: np.ndarray, prediction_threshold: float
    ) -> None:
        """Construct the Evaluation object
        :y_true: y_true (array-like, shape (n_samples)) – Ground truth (correct) target values.
        :y_proba: (array-like, shape (n_samples, 2)) – Prediction probabilities for the two classes
            returned by a classifier.
        :prediction_threshold; Threshold over which to predict a
        :return: None
        """

        if not (0 < prediction_threshold < 1):
            raise ValueError("prediction_threshold needs to be between 0 and 1.")

        if len(y_true) != len(y_proba):
            raise ValueError("Length of y_true and y_proba must be the same.")
        self.y_true=y_true
        self.y_proba = y_proba
        self.prediction_threshold = prediction_threshold
        self.y_pred = (self.y_proba[:, 1] > self.prediction_threshold)

    def get_classification_report(self) -> dict:
        return classification_report(
            self.y_true,
            self.y_pred,
            output_dict=True,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
        )

    def plot_auc(self, outpath: Path) -> None:
        """Plot AUC curve
        The plot is saved to outpath
        :outpath: Outpath for plot.
        :return: None
        """
        fig, ax = plt.subplots()
        _ = skplt.metrics.plot_roc(
            self.y_true,
            self.y_proba,
            plot_micro=False,
            plot_macro=False,
            classes_to_plot=[1],
            title="ROC Curve for SUFFL=1",
        )
        plt.savefig(str(outpath))
        plt.close()

    def plot_probability_calibration_curve(self, outpath: Path) -> None:
        """Plot probability calibration curve
        The plot is saved to outpath
        :return: None
        """
        fig, ax = plt.subplots()
        _ = skplt.metrics.plot_calibration_curve(
            self.y_true, probas_list=[self.y_proba], title="Calibration plot."
        )
        plt.savefig(str(outpath))
        plt.close()

    def plot_precision_recall(self, outpath: Path) -> None:
        """Plot precision-recalls curve
        The plot is saved to outpath
        :outpath: Outpath for plot.
        :return: None
        """
        fig, ax = plt.subplots()
        _ = skplt.metrics.plot_precision_recall(
            self.y_true, self.y_proba, plot_micro=False, classes_to_plot=[1], title="Precision-recall Curve"
        )
        plt.savefig(str(outpath))
        plt.close()

    def save_evaluation_artifacts(self, outdir: Path) -> None:
        """Save all evaluation artifacts to a folder"""
        self.plot_auc(outdir / Path("auc_plot.png"))
        self.plot_precision_recall(outdir / Path("precision_recall_plot.png"))
        self.plot_probability_calibration_curve(outdir / Path("probability_calibration_plot.png"))
        with open(outdir / Path("metrics.json"), "w") as f:
            json.dump(self.get_classification_report(), f)
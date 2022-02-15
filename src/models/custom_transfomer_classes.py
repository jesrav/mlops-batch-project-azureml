"""Module that contains custom sklearn compatible transformer classes."""
from typing import List

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a subset of columns from a pandas dataframe
    """
    def __init__(self, columns: List[str]):
        """Constructor method

        :param pandera_schema: Pandera schema
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.columns]

    def get_params(self, deep=True):
        return {
            "columns": self.columns,
        }

    def set_params(self, **kwargs):
        if "columns" in kwargs:
            setattr(self, "columns", kwargs["columns"])

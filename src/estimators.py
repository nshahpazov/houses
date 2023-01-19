"""Helper classes for transforming datasets in some way"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Pandalizer(BaseEstimator, TransformerMixin):
    """
    Executes a transformer on a subset of columns and returns a
    Pandas DataFrame as a result
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self.columns = None

    def fit(self, df, y=None, columns=None, **fit_args):
        """Fit the Pandalizer taking additional fit_args to pass"""
        # get only a subset of columns on which to apply the transformer
        if columns is None:
            self.columns = df.columns
        else:
            self.columns = np.intersect1d(columns, df.columns)
        return self.transformer.fit(df[self.columns], y, **fit_args)

    def transform(self, data):
        """Transform the data"""
        df = data.copy()
        df.loc[:, self.columns] = self.transformer.transform(df[self.columns])
        return df

    def fit_transform(self, X, y=None, columns=None, **fit_args):
        """Fit_Transform the data"""
        # pylint: disable=unused-argument
        self.fit(X, y, columns, **fit_args)
        return self.transform(X)

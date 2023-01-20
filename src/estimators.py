"""Helper classes for transforming datasets in some way"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class RareCategoriesReplacer(BaseEstimator, TransformerMixin):
    """
    Replaces Categorical Columns rare values with a keyword
    """
    def __init__(self, threshold=0.05, keyword: str='Other') -> None:
        self.keyword = keyword
        self.threshold = threshold
        self.proportions = []


    def get_proportions(self, X):
        """
        Get the proportions of the keywords in the categorical columns
        """
        counts = [pd.Series(x).value_counts(normalize=True) for x in X.T]
        return counts


    def fit(self, X, y=None):
        """Fit the rare categorical transformer"""
        # pylint: disable=unused-argument
        is_df = isinstance(X, pd.DataFrame)
        self.proportions = self.get_proportions(X.values if is_df else X)
        return self


    def is_to_replace(self, i, col):
        """calculate keywords to replace by a given column"""
        props = self.proportions[i]
        rares = props[props < self.threshold].index.values
        new_ones = np.setdiff1d(col, props.index)
        is_rare = np.isin(col, rares)
        is_new_one = np.isin(col, new_ones)
        return is_rare | is_new_one


    def transform(self, X) -> pd.DataFrame:
        """Transform the rare categorical transformer"""
        is_df = isinstance(X, pd.DataFrame)
        Xt = X.values.copy() if is_df else X.values.copy()
        for i, col in enumerate(Xt.T):
            col[self.is_to_replace(i, col)] = self.keyword
        return Xt


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

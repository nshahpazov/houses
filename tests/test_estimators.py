"""Testing custom estimators module"""
import pandas as pd

from houses.estimators import RareCategoriesReplacer, Pandalizer


def test_rare_categories_replacer_big_threshold():
    rare_categories_replacer = RareCategoriesReplacer(threshold=0.90)
    X = pd.DataFrame({"category": pd.Categorical(["a", "b", "c", "a", "a", "a"])})
    Xt = rare_categories_replacer.fit_transform(X)
    all_are_other = (Xt[:, 0] == "Other").all()
    assert all_are_other


def test_rare_categories_replacer_small_threshold():
    rare_categories_replacer = RareCategoriesReplacer(threshold=0.1, keyword="Arnold")
    X = pd.DataFrame({"category": pd.Categorical(list("abbcdffeacdg"))})
    X.value_counts(normalize=True)
    # we expect it to replace e and g, i.e. they are not in the end result
    Xt = rare_categories_replacer.fit_transform(X)
    Xt_df = pd.DataFrame(Xt, columns=["category"])
    Xt_df["category"].isin(["e", "g"])
    has_letters = Xt_df["category"].isin(["e", "g"]).any()
    assert not has_letters

    is_letters = X["category"].isin(["e", "g"])
    is_rare = Xt_df["category"] == "Arnold"
    # indices match
    has_replaced = (X[is_letters].index == Xt_df[is_rare].index).all()

    assert has_replaced


def test_pandalizer():
    rare_replacer = RareCategoriesReplacer()
    dataframe = pd.DataFrame({"x": pd.Categorical(list("abcdddaabbccfggyjhhuj"))})
    pandalizer = Pandalizer(rare_replacer)
    transformed_df = pandalizer.fit_transform(dataframe)
    assert transformed_df.columns == dataframe.columns
    assert isinstance(transformed_df, pd.DataFrame)

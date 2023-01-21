"""Evaluation module"""
import matplotlib.pyplot as plt
import pickle
from sklearn.inspection import permutation_importance
from pathlib import Path
from typing import Union
import click
import constants as constants
import pandas as pd
from plotnine import (ggplot, aes, xlab, ylab, ggtitle, geom_bar, coord_flip, geom_boxplot, theme)
import seaborn as sns
import numpy as np

@click.command()
@click.argument(
    'model_input_path',
    type=click.Path(exists=True),
    default=constants.DEFAULT_MODEL_PATH,
)
@click.argument(
    'x_train_filepath',
    type=click.Path(exists=True),
    default=constants.DEFAULT_X_TRAIN_PATH,
)
@click.argument(
    'y_train_filepath',
    type=click.Path(exists=True),
    default=constants.DEFAULT_Y_TRAIN_PATH,
)
@click.argument(
    'x_test_filepath',
    type=click.Path(exists=True),
    default=constants.DEFAULT_X_TEST_PATH,
)
@click.argument(
    'y_test_filepath',
    type=click.Path(exists=True),
    default=constants.DEFAULT_Y_TEST_PATH,
)
def evaluate(
    model_input_path: Union[str, Path],
    x_train_filepath: Union[str, Path],
    y_train_filepath: Union[str, Path],
    x_test_filepath: Union[str, Path],
    y_test_filepath: Union[str, Path],
):
    # load model
    with open(model_input_path, "rb") as openfile:
        model_pipeline = pickle.load(openfile)
    
    # load train and test feature sets + ground truths
    y_train = pd.read_csv(y_train_filepath).values
    y_test = pd.read_csv(y_test_filepath).values

    X_train = pd.read_csv(x_train_filepath)
    X_test = pd.read_csv(x_test_filepath)

    # Report training set score
    train_score = model_pipeline.score(X_train, y_train) * 100
    test_score = model_pipeline.score(X_test, y_test) * 100

    # Write scores to a file
    with open("data/model_output/metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)

    # feature importance
    importances_plot = generate_feature_importance_plot(model_pipeline, X_test, y_test)
    importances_plot.save("data/model_output/permutation_feature_importance.png")


def generate_feature_importance_plot(model, X_val: pd.DataFrame, y_val: pd.Series):
    scoring = ['neg_mean_squared_error']
    r_multi = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=0, scoring=scoring,
    )
    importances_df = pd.DataFrame(
        r_multi["neg_mean_squared_error"]["importances"].T,
        columns=X_val.columns,
    )

    mean_importances_df = pd.DataFrame({
        "importance": r_multi["neg_mean_squared_error"]["importances_mean"],
        "variable": X_val.columns,
    })

    above_0_importances_df = mean_importances_df[mean_importances_df["importance"] > 1000]
    above_0_importance_columns = above_0_importances_df["variable"]

    importances_plot = (importances_df[above_0_importance_columns]
     .melt()
     .pipe(ggplot) +
        aes(x="variable", y="value") +
        geom_boxplot() +
        coord_flip() +
        theme(figure_size=(6, 22))
    )
    return importances_plot


if __name__ == "__main__":
    evaluate()
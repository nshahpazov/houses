"""Train module"""
import pathlib
import pickle
from typing import Union

import click
import constants
import pandas as pd
from estimators import RareCategoriesReplacer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer

# modeling
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

__ordinal_encoder = OrdinalEncoder(
    categories=[constants.ORDINALS_ORDERING] * len(constants.ORDINALS),
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)


@click.command()
@click.argument(
    "input_x_filepath",
    type=click.Path(exists=True),
    default=constants.DEFAULT_X_TRAIN_PATH,
)
@click.argument(
    "input_y_filepath",
    type=click.Path(exists=True),
    default=constants.DEFAULT_Y_TRAIN_PATH,
)
@click.argument(
    "model_output_filepath",
    type=click.Path(),
    default=constants.DEFAULT_MODEL_PATH,
)
@click.option(
    "--model_seed",
    "model_seed",
    required=False,
    default=constants.DEFAULT_MODEL_SEED,
    help="Regularization model seed",
)
@click.option(
    "--rare_threshold",
    "rare_threshold",
    required=False,
    default=0.05,
    help="Rare Threshold",
)
def train(
    input_x_filepath: Union[str, pathlib.Path],
    input_y_filepath: Union[str, pathlib.Path],
    model_output_filepath: Union[str, pathlib.Path],
    model_seed: int,
    rare_threshold: float,
) -> None:
    X_train = pd.read_csv(input_x_filepath)
    y_train = pd.read_csv(input_y_filepath)
    # all kind of steps. cross-validation, regularization, feature selection, hyperparameter tuning
    # the end result is a model on the entire train set we have

    categoricals = X_train.columns.intersection(constants.CATEGORICAL_COLUMNS)
    numericals = X_train.columns.intersection(constants.NUMERICAL_COLUMNS)

    train_pipeline = create_lasso_pipeline(
        rare_threshold=rare_threshold,
        categoricals=categoricals,
        numericals=numericals,
        # TODO: parameterize this as well
        alpha=0,
        model_seed=model_seed,
    )

    # fit the train pipeline
    train_pipeline.fit(X=X_train, y=y_train)

    # save the model and the final predictions
    with open(model_output_filepath, "wb") as handle:
        pickle.dump(train_pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_lasso_pipeline(rare_threshold, categoricals, numericals, alpha, model_seed):
    """Create a pipeline for the lasso regression"""

    numeric_pipeline = Pipeline(
        [
            ("impute_with_mean", SimpleImputer(strategy="mean")),
            ("scaling", StandardScaler()),
            ("yeo_johnson", PowerTransformer()),
        ]
    )

    # transformations to be applied to categorical features
    categoric_pipeline = Pipeline(
        [
            ("replace_rare", RareCategoriesReplacer(rare_threshold)),
            ("one_hot_encoding", OneHotEncoder()),
        ]
    )

    # transformations on the response variable
    response_variable_pipeline = Pipeline(
        [
            ("normalization", StandardScaler()),
            ("yeo_johnson", PowerTransformer()),
        ]
    )

    all_transformations = ColumnTransformer(
        transformers=[
            ("numeriric_transformations", numeric_pipeline, numericals),
            ("categoric_transformations", categoric_pipeline, categoricals),
            ("ordinals_encoding", __ordinal_encoder, constants.ORDINALS),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("column_transformations", all_transformations),
            (
                "lasso_and_target_transform",
                TransformedTargetRegressor(
                    regressor=LinearRegression(),
                    transformer=response_variable_pipeline,
                ),
            ),
        ]
    )
    return pipeline


if __name__ == "__main__":
    train()

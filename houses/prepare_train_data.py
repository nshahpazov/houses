"""Split module"""
import pathlib
import typing

import click
import constants
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument(
    "input_filepath",
    type=click.Path(exists=True),
    default=constants.DEFAULT_TRAIN_INPUT_PATH,
)
@click.option(
    "--split_seed",
    "split_seed",
    required=False,
    default=1,
    help="Splitting seed",
)
@click.option(
    "--train_size",
    "train_size",
    required=False,
    default=constants.DEFAULT_TRAIN_SET_SIZE,
    help="Train set size to be used for the training",
)
def prepare(
    input_filepath: typing.Union[str, pathlib.Path],
    split_seed: int,
    train_size: float,
) -> None:
    """
    Main command function to be executed by the script
    """
    input_df = pd.read_csv(input_filepath)
    X = input_df.drop(columns=[constants.TARGET_VARIABLE_NAME, constants.ID_COLUMN])
    y = input_df[constants.TARGET_VARIABLE_NAME]

    is_to_split = train_size < 1
    if is_to_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=split_seed
        )
        # TODO: use parameterized filenames
        X_train.to_csv("data/model_input/x_train.csv", index=False)
        X_test.to_csv("data/model_input/x_test.csv", index=False)
        y_train.to_csv("data/model_input/y_train.csv", index=False)
        y_test.to_csv("data/model_input/y_test.csv", index=False)
    else:
        X.to_csv("data/model_input/x_train_final.csv", index=False)
        y.to_csv("data/model_input/y_train_final.csv", index=False)


if __name__ == "__main__":
    prepare()

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import numpy
import pandas

from concrete.ml.sklearn import XGBClassifier as ConcreteXGBoostClassifier

# Max Input to be displayed on the HuggingFace space brower using Gradio
# Too large inputs, slow down the server: https://github.com/gradio-app/gradio/issues/1877
INPUT_BROWSER_LIMIT = 380

# Store the server's URL
SERVER_URL = "http://localhost:8000/"

CURRENT_DIR = Path(__file__).parent
DEPLOYMENT_DIR = CURRENT_DIR / "deployment_files"
KEYS_DIR = DEPLOYMENT_DIR / ".fhe_keys"
CLIENT_DIR = DEPLOYMENT_DIR / "client_dir"
SERVER_DIR = DEPLOYMENT_DIR / "server_dir"

ALL_DIRS = [KEYS_DIR, CLIENT_DIR, SERVER_DIR]

# Columns that define the target
TARGET_COLUMNS = ["prognosis_encoded", "prognosis"]

TRAINING_FILENAME = "./data/Training_preprocessed.csv"
TESTING_FILENAME = "./data/Testing_preprocessed.csv"

# pylint: disable=invalid-name


def pretty_print(
    inputs, case_conversion=str.title, which_replace: str = "_", to_what: str = " ", delimiter=None
):
    """
    Prettify and sort the input as a list of string.

    Args:
        inputs (Any): The inputs to be prettified.

    Returns:
        List: The prettified and sorted list of inputs.

    """
    # Flatten the list if required
    pretty_list = []
    for item in inputs:
        if isinstance(item, list):
            pretty_list.extend(item)
        else:
            pretty_list.append(item)

    # Sort
    pretty_list = sorted(list(set(pretty_list)))
    # Replace
    pretty_list = [item.replace(which_replace, to_what) for item in pretty_list]
    pretty_list = [case_conversion(item) for item in pretty_list]
    if delimiter:
        pretty_list = f"{delimiter.join(pretty_list)}."

    return pretty_list


def clean_directory() -> None:
    """
    Clear direcgtories
    """
    print("Cleaning...\n")
    for target_dir in ALL_DIRS:
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)


def get_disease_name(encoded_prediction: int, file_name: str = TRAINING_FILENAME) -> str:
    """Return the disease name given its encoded label.

    Args:
        encoded_prediction (int): The encoded prediction
        file_name (str): The data file path

    Returns:
        str: The according disease name
    """
    df = pandas.read_csv(file_name, usecols=TARGET_COLUMNS).drop_duplicates()
    disease_name, _ = df[df[TARGET_COLUMNS[0]] == encoded_prediction].values.flatten()
    return disease_name


def load_data() -> Union[Tuple[pandas.DataFrame, numpy.ndarray], List]:
    """
    Return the data

    Args:
        None

    Return:
        The train, testing set and valid symptoms.
    """
    # Load data
    df_train = pandas.read_csv(TRAINING_FILENAME)
    df_test = pandas.read_csv(TESTING_FILENAME)

    # Separate the traget from the training / testing set:
    # TARGET_COLUMNS[0] -> "prognosis_encoded" -> contains the numeric label of the disease
    # TARGET_COLUMNS[1] -> "prognosis"         -> contains the name of the disease

    y_train = df_train[TARGET_COLUMNS[0]]
    X_train = df_train.drop(columns=TARGET_COLUMNS, axis=1, errors="ignore")

    y_test = df_test[TARGET_COLUMNS[0]]
    X_test = df_test.drop(columns=TARGET_COLUMNS, axis=1, errors="ignore")

    return (
        (X_train, X_test),
        (y_train, y_test),
        X_train.columns.to_list(),
        df_train[TARGET_COLUMNS[1]].unique().tolist(),
    )


def load_model(X_train: pandas.DataFrame, y_train: numpy.ndarray):
    """
    Load a pre-trained serialized model

    Args:
        X_train (pandas.DataFrame): Training set
        y_train (numpy.ndarray): Targets of the training set

    Return:
        The Concrete ML model and its circuit
    """
    # Parameters
    concrete_args = {"max_depth": 1, "n_bits": 3, "n_estimators": 3, "n_jobs": -1}
    classifier = ConcreteXGBoostClassifier(**concrete_args)
    # Train the model
    classifier.fit(X_train, y_train)
    # Compile the model
    circuit = classifier.compile(X_train)

    return classifier, circuit

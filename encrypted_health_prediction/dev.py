"""Generating deployment files."""

import shutil

from pathlib import Path

import pandas as pd

from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.deployment import FHEModelDev


# Data files location
TRAINING_FILE_NAME = "./data/Training_preprocessed.csv"
TESTING_FILE_NAME = "./data/Testing_preprocessed.csv"

# Load data
df_train = pd.read_csv(TRAINING_FILE_NAME)
df_test = pd.read_csv(TESTING_FILE_NAME)

# Split the data into X_train, y_train, X_test_, y_test sets
TARGET_COLUMN = ["prognosis_encoded", "prognosis"]

y_train = df_train[TARGET_COLUMN[0]].values.flatten()
y_test = df_test[TARGET_COLUMN[0]].values.flatten()

X_train = df_train.drop(TARGET_COLUMN, axis=1)
X_test = df_test.drop(TARGET_COLUMN, axis=1)

# Concrete ML model

# Models parameters
optimal_param = {"C": 0.9, "n_bits": 13, "solver": "sag", "multi_class": "auto"}

clf = ConcreteLogisticRegression(**optimal_param)

# Fit the model
clf.fit(X_train, y_train)

# Compile the model
fhe_circuit = clf.compile(X_train)

fhe_circuit.client.keygen(force=False)

path_to_model = Path("./deployment_files/").resolve()

if path_to_model.exists():
    shutil.rmtree(path_to_model)

dev = FHEModelDev(path_to_model, clf)

dev.save(via_mlir=True)
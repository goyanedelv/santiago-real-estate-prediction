# 0. Preamble
## Libraries
import pandas as pd
import numpy as np
import yaml

## Model modules
from models import *

## ML stuff
from sklearn.model_selection import train_test_split

## YAML for config
with open("config.yaml", 'r') as stream:
    try:
        parameters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# 1. Load data and minor pre-process
raw = pd.read_csv("data/master_table.csv")

data = raw.dropna(subset=["avg_UF_per_m2"])
data = data[data["NOM_PROVIN"] == "SANTIAGO"]

manzana = data["manzana"]
y = np.log(data["avg_UF_per_m2"])

cols_2_drop = ["COMUNA_y", "HOMBRES", "MUJERES", "EDAD_0A5", "avg_UF_per_m2",
                "EDAD_6A14", "EDAD_15A64", "EDAD_65YMAS", "NOM_PROVIN",
                "INMIGRANTES", "PUEBLO", "ID_ZONA_LOC.1"]

data = data.drop(cols_2_drop, axis=1)

data = pd.get_dummies(data, prefix = ["NOM_COMUNA"])

## Polynomials
data["energy_sq"] = data["energy"]**2
data["green_sq"] = data["green"]**2
data["correlation_sq"] = data["correlation"]**2
data["homogeneity_sq"] = data["homogeneity"]**2
data["dissimilarity_sq"] = data["dissimilarity"]**2
data["lat_sq"] = data["lat"]**2
data["lon_sq"] = data["lon"]**2

# 2. Split train-test
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = parameters["test_size"], random_state = parameters["random_state"])

manzana_train = X_train["manzana"]
X_train = X_train.drop(["manzana"], axis=1)

manzana_test = X_test["manzana"]
X_test = X_test.drop(["manzana"], axis=1)

# 3. Model training & outputs
if parameters["xgboost"]:
    print("Running XGBoost")
    run_xgboost(parameters, X_train, y_train, X_test, y_test)

if parameters["rf"]:
    print("Running Random Forest")
    run_rf(parameters, X_train, y_train, X_test, y_test)

if parameters["svm"]:
    print("Running Support Vector Machines")
    run_svm(parameters, X_train, y_train, X_test, y_test)

if parameters["neuralnetwork"]:
    print("Running Neural Networks")
    run_nnet(parameters, X_train, y_train, X_test, y_test)

if parameters["glm"]:
    print("Running Generalized Linear Model")
    run_glm(parameters, X_train, y_train, X_test, y_test)

if parameters["lasso"]:
    print("Running Lasso Regression")
    run_lasso(parameters, X_train, y_train, X_test, y_test)

if parameters["ridge"]:
    print("Running Ridge Regression")
    run_ridge(parameters, X_train, y_train, X_test, y_test)
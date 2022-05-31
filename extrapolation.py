# 0. Preamble
import pandas as pd
import numpy as np
import yaml
from models import *

with open("config.yaml", 'r') as stream:
    try:
        parameters = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# 1. Load data and pre-process
raw = pd.read_csv("data/master_table.csv")
raw = raw[raw["NOM_PROVIN"] == "SANTIAGO"]

# Separate data with price and without price
data_train = raw[raw["avg_UF_per_m2"] > 0 ]

data_extrapolate = raw[pd.isnull(raw["avg_UF_per_m2"]) ]

manzana_train = data_train["manzana"]
y_train = data_train["avg_UF_per_m2"]
comuna_train = data_train["NOM_COMUNA"]

manzana_xp = data_extrapolate["manzana"]
#y_xp = data_extrapolate["avg_UF_per_m2"]
comuna_xp = data_extrapolate["NOM_COMUNA"]


cols_2_drop = ["COMUNA_y", "HOMBRES", "MUJERES", "EDAD_0A5", "avg_UF_per_m2",
                "EDAD_6A14", "EDAD_15A64", "EDAD_65YMAS", "NOM_PROVIN",
                "INMIGRANTES", "PUEBLO", "ID_ZONA_LOC.1", "NOM_COMUNA", "manzana"]

data_train = data_train.drop(cols_2_drop, axis=1)
data_extrapolate = data_extrapolate.drop(cols_2_drop, axis=1)

## Polynomials
data_train["energy_sq"] = data_train["energy"]**2
data_train["green_sq"] = data_train["green"]**2
data_train["correlation_sq"] = data_train["correlation"]**2
data_train["homogeneity_sq"] = data_train["homogeneity"]**2
data_train["dissimilarity_sq"] = data_train["dissimilarity"]**2
data_train["lat_sq"] = data_train["lat"]**2
data_train["lon_sq"] = data_train["lon"]**2

data_extrapolate["energy_sq"] = data_extrapolate["energy"]**2
data_extrapolate["green_sq"] = data_extrapolate["green"]**2
data_extrapolate["correlation_sq"] = data_extrapolate["correlation"]**2
data_extrapolate["homogeneity_sq"] = data_extrapolate["homogeneity"]**2
data_extrapolate["dissimilarity_sq"] = data_extrapolate["dissimilarity"]**2
data_extrapolate["lat_sq"] = data_extrapolate["lat"]**2
data_extrapolate["lon_sq"] = data_extrapolate["lon"]**2

best_parameters = parameters["best_model"]

regr = GradientBoostingRegressor(**best_parameters, random_state = parameters["random_state"])
regr.fit(data_train, np.log(y_train))

data_extrapolate = data_extrapolate.dropna()

y_xp = regr.predict(data_extrapolate)
y_train_pred = regr.predict(data_train)

data_extrapolate["price"] = np.exp(y_xp.copy())
data_train["price"] = np.exp(y_train_pred.copy())
data_train["actuals"] = y_train.copy()
data_extrapolate["actuals"] = -1
data_train["manzana"] = manzana_train
data_extrapolate["manzana"] = manzana_xp
data_train["comuna"] = comuna_train
data_extrapolate["comuna"] = comuna_xp

full_df = data_extrapolate.append(data_train)

full_df.to_excel("data/predicted_and_actuals_full_df.xlsx", index = False)
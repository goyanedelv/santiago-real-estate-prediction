import pandas as pd
import numpy as np

# Loading block level census and slicing only the Santiago data
data_block = pd.read_csv("02_data_censo/Censo2017_Manzanas.csv", sep = ";")

#data_block["ID_MANZENT"] = data_block["ID_MANZENT"].astype(str)
data_block_stgo = data_block[data_block["REGION"] == 13]
del data_block

# Loading person level census and slicing only Santiago data
data_person = pd.read_csv("02_data_censo/censo_personas/Microdato_Censo2017-Personas.csv", sep = ";")
data_person_stgo = data_person[data_person["REGION"] == 13]
del data_person

## ESCOLARIDAD (Schooling years)
### 98 and 99 --> nan
data_person_stgo.loc[data_person_stgo["ESCOLARIDAD"] == 98,"ESCOLARIDAD"] = np.nan
data_person_stgo.loc[data_person_stgo["ESCOLARIDAD"] == 99,"ESCOLARIDAD"] = np.nan

## P17 (EMPLOYMENT)
### 98 and 99 --> 8
data_person_stgo.loc[data_person_stgo["P17"] == 98,"P17"] = 8
data_person_stgo.loc[data_person_stgo["P17"] == 99,"P17"] = 8

# Dummify employment
data_person_stgo = pd.get_dummies(data_person_stgo, columns = ["P17"])

# Aggregate and group by zone
data_person_stgo_agg = data_person_stgo.groupby("ID_ZONA_LOC").aggregate({"ESCOLARIDAD": ["mean"],
                                                                         "P17_1" : ["sum"],
                                                                         "P17_2" : ["sum"],                                                                         
                                                                         "P17_3" : ["sum"],
                                                                         "P17_4" : ["sum"],
                                                                         "P17_5" : ["sum"],
                                                                         "P17_6" : ["sum"],
                                                                         "P17_7" : ["sum"],
                                                                         "P17_8" : ["sum"],
                                                                         "PERSONAN": ["count"]}).reset_index()

# Rename columns (avoid multi-index)
keep_this_cols = ["ID_ZONA_LOC", "ESCOLARIDAD", "P17_1", "P17_2", "P17_3", "P17_4", "P17_5", "P17_6", "P17_7", "P17_8", "PERSONAN"]

data_person_stgo_agg.columns = keep_this_cols

data_person_stgo_agg["P17_1"] = data_person_stgo_agg["P17_1"] / data_person_stgo_agg["PERSONAN"]
data_person_stgo_agg["P17_2"] = data_person_stgo_agg["P17_2"] / data_person_stgo_agg["PERSONAN"]
data_person_stgo_agg["P17_3"] = data_person_stgo_agg["P17_3"] / data_person_stgo_agg["PERSONAN"]
data_person_stgo_agg["P17_4"] = data_person_stgo_agg["P17_4"] / data_person_stgo_agg["PERSONAN"]
data_person_stgo_agg["P17_5"] = data_person_stgo_agg["P17_5"] / data_person_stgo_agg["PERSONAN"]
data_person_stgo_agg["P17_6"] = data_person_stgo_agg["P17_6"] / data_person_stgo_agg["PERSONAN"]
data_person_stgo_agg["P17_7"] = data_person_stgo_agg["P17_7"] / data_person_stgo_agg["PERSONAN"]
data_person_stgo_agg["P17_8"] = data_person_stgo_agg["P17_8"] / data_person_stgo_agg["PERSONAN"]

data_person_stgo_agg = data_person_stgo_agg.drop("PERSONAN", axis = 1)

del data_person_stgo

# Merge with census at block level
data_block_stgo_2 = data_block_stgo.merge(data_person_stgo_agg, how = "left", on = "ID_ZONA_LOC")


keep_this_cols_2 = ['COMUNA',
       'ID_ZONA_LOC', 'ID_MANZENT', 'PERSONAS', 'HOMBRES', 'MUJERES',
       'EDAD_0A5', 'EDAD_6A14', 'EDAD_15A64', 'EDAD_65YMAS', 'INMIGRANTES',
       'PUEBLO', 'VIV_PART', 'VIV_COL', 'VPOMP', 'TOTAL_VIV', 'CANT_HOG',
       'P01_1', 'P01_2', 'P01_3', 'P01_4', 'P01_5', 'P01_6', 'P01_7', 'P03A_1',
       'P03A_2', 'P03A_3', 'P03A_4', 'P03A_5', 'P03A_6', 'P03B_1', 'P03B_2',
       'P03B_3', 'P03B_4', 'P03B_5', 'P03B_6', 'P03B_7', 'P03C_1', 'P03C_2',
       'P03C_3', 'P03C_4', 'P03C_5', 'MATACEP', 'MATREC', 'MATIRREC', 'P05_1',
       'P05_2']

data_block_stgo_3 = data_block_stgo_2[keep_this_cols_2 + keep_this_cols[:-1]]

data_block_stgo_3.to_excel("09_add_census_data/census_data_at_block_level.xlsx", index = False)
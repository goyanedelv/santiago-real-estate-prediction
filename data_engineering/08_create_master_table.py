import pandas as pd
import numpy as np

# Paths
path_ft = "10_feature_extraction/features_of_img_blocks.xlsx"
path_census = "09_add_census_data/census_data_at_block_level.xlsx"
path_prices_and_blocks = "08_aggregate_by_block/blocks_with_prices.xlsx"

# Read
img_fts = pd.read_excel(path_ft, engine='openpyxl')
census = pd.read_excel(path_census, engine='openpyxl')
prices = pd.read_excel(path_prices_and_blocks, engine='openpyxl')

# See columns
img_fts.columns
census.columns
prices.columns

# homogeneize columns: manzana (block in spanish)
census.rename({'ID_MANZENT': 'manzana'}, axis=1, inplace=True)
prices.rename({'MANZENT_I': 'manzana'}, axis=1, inplace=True)

# merging datasets
master_table = prices.merge(img_fts, how = "left", on = "manzana")
master_table_2 = master_table.merge(census, how = "left", on = "manzana")

# See columns
master_table_2.columns

# Dropping unuseful columns (broader political division)
cols_2_drop = ['REGION', 'NOM_REGION', 'PROVINCIA', 'COMUNA_x',
                'DISTRITO', 'LOC_ZON', 'ENT_MAN', 'CATEGORIA',
                'NOM_CATEGO', "REGION", "NOM_REGION",
                "DISTRITO", "LOC_ZON", "ENT_MAN", "CATEGORIA",
                "NOM_CATEGO", "ID_ZONA_LOC", "PERSONAS", "TOTAL_VIV"]
master_table_2 = master_table_2.drop(cols_2_drop, axis=1)

# Transforming coords to lat and lon
coords = master_table_2["coords"].to_list()
lat_lst = []
lon_lst = []

for i in range(len(coords)):
    pos = coords[i].find(",")
    lon = float(coords[i][1:pos])
    lat = float(coords[i][pos+1:-1])
    
    lat_lst.append(lat)
    lon_lst.append(lon)

master_table_2["lat"] = lat_lst
master_table_2["lon"] = lon_lst

master_table_2.drop('coords', axis=1, inplace=True)

# Checking everything makes sense
master_table_2.loc[master_table_2["NOM_COMUNA"] == "VITACURA", "avg_UF_per_m2"].mean()
master_table_2.loc[master_table_2["NOM_COMUNA"] == "LAS CONDES", "avg_UF_per_m2"].mean()
master_table_2.loc[master_table_2["NOM_COMUNA"] == "PIRQUE", "avg_UF_per_m2"].mean()
master_table_2.loc[master_table_2["NOM_COMUNA"] == "SANTIAGO", "avg_UF_per_m2"].mean()
master_table_2.loc[master_table_2["NOM_COMUNA"] == "LA PINTANA", "avg_UF_per_m2"].mean()

# remove the outliers
# if numeric
master_table_2.loc[master_table_2["avg_UF_per_m2"] < 9, "avg_UF_per_m2"] = np.nan
master_table_2.loc[master_table_2["avg_UF_per_m2"] > 600, "avg_UF_per_m2"] = np.nan

master_table_2["avg_UF_per_m2"].describe()

# Saving the master data table as csv
master_table_2.to_csv("11_master_data_table/master_table.csv", index = False, encoding='utf-8-sig')
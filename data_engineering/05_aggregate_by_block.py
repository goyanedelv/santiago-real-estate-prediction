from cmath import inf
import pandas as pd
import numpy as np
import time

blocks_raw = pd.read_excel("06_satellites images/geolocated_polygons.xlsx", engine='openpyxl')
blocks = blocks_raw[blocks_raw['coords'] != '(None, None)']
blocks['MANZENT_I'] = blocks['MANZENT_I'].astype(str)

listings_raw = pd.read_excel("07_block_allocation/all_listings_with_manzana.xlsx", engine='openpyxl')
listings_raw['manzana'] = listings_raw['manzana'].astype(str)

listings_raw['surface'] = listings_raw['surface'].astype(str)
listings_raw['surface'] = listings_raw['surface'].apply(lambda x: x.replace("\n","").replace(",","."))
listings_raw['surface'] = listings_raw['surface'].astype(float)


listings_raw['UF_per_m2'] = listings_raw['price'] / listings_raw['surface']

df_1 = listings_raw[np.isfinite(listings_raw['UF_per_m2'])] # values without infinite
df_2 = listings_raw[~np.isfinite(listings_raw['UF_per_m2'])] # values with infinite

agg_data = df_1.groupby(['manzana']).agg(avg_UF_per_m2=('UF_per_m2', 'mean'))

blocks = blocks.merge(agg_data, how='left', left_on='MANZENT_I', right_on='manzana')

blocks[blocks['avg_UF_per_m2'].notnull()].shape # looking good

blocks.to_excel("08_aggregate_by_block/blocks_with_prices.xlsx", index = False)
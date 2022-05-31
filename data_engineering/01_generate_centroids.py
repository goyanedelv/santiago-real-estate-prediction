# import shapefile
import geojson
from shapely.geometry import Polygon, Point, mapping, shape
import matplotlib.pyplot as plt
import pandas as pd

filename = "chile_geojson/manzanas/R13_MANZANA_IND_C17.shp.geojson"

with open(filename, encoding="utf8") as f:
    gj = geojson.load(f)


full_lenn = len(gj['features'])

latlon = []
charac = []

for i in range(full_lenn):
    features = gj['features'][i]
    additional = features['properties'].values()

    coords = features['geometry']['coordinates'][0]
    if len(coords) > 2:
        coords_as_pol = Polygon(coords)
        tuple_ = list(coords_as_pol.centroid.coords)[0]
    else:
        tuple_ = (None, None)

    charac.append(additional)
    latlon.append(tuple_)

df1 = pd.DataFrame({'coords': latlon})

df2 = pd.DataFrame(charac)

cols_df2 = list(features['properties'].keys())

df2.columns = cols_df2

df3 = pd.concat([df1, df2], axis=1)

df3.to_excel("geolocated_polygons.xlsx", index = False)
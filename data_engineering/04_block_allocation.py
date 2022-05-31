import pandas as pd
import time

blocks_raw = pd.read_excel("06_satellites images/geolocated_polygons.xlsx", engine='openpyxl')
blocks = blocks_raw[blocks_raw['coords'] != '(None, None)']

listings_raw = pd.read_excel("05_geocoding/all_listings.xlsx", engine='openpyxl')
listings = listings_raw[listings_raw['geocode'] != '(None, None)']

def coord_as_string(string):
    pos = string.find(",")

    lon = float(string[1:pos])
    lat = float(string[pos+1:-1])

    return lat, lon

block_coords = blocks['coords'].to_list()

listings_coords = listings['geocode'].to_list()

block_coords_2 = [coord_as_string(x) for x in block_coords]

listings_coords_2 = [coord_as_string(x) for x in listings_coords]

manzanas = blocks['MANZENT_I'].to_list()


# for every listing I want to know its closest block, and then assign it!
db = []
for i in listings_coords_2:
    distance = list(map(lambda y: ((i[0] - y[1])**2 + (i[1] - y[0])**2)**(1/2), block_coords_2))
    
    minimal = min(distance)
    indice = distance.index(minimal)

    pack = [minimal, manzanas[indice]]

    db.append(pack)


dist_manzana = pd.DataFrame(db, columns = ['distance', 'manzana'])


# listings_with_manzana = pd.concat([listings, dist_manzana], axis=1)
listings['dist'] = dist_manzana['distance'].to_list()
listings['manzana'] = dist_manzana['manzana'].to_list()

listings.to_excel("07_block_allocation/all_listings_with_manzana.xlsx", index = False)

# re run but avoiding the missings!
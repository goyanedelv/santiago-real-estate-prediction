import requests
import pandas as pd

data = pd.read_excel("geolocated_polygons.xlsx", engine='openpyxl')

data_exc_none = data[data['coords'] != '(None, None)']

# Enter your api key here
api_key = "53CR37"
  
# url variable store url
url = "https://maps.googleapis.com/maps/api/staticmap?"

manzana = data_exc_none['MANZENT_I'].to_list()
coords = data_exc_none['coords'].to_list()

big_lenn = len(manzana)

zoom = 17
map_type = 'satellite'


for i in range(31206, big_lenn):

    pos = coords[i].find(",")

    lon = float(coords[i][1:pos])
    lat = float(coords[i][pos+1:-1])
    link = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=300x300&maptype={map_type}&key={api_key}"

    r = requests.get(link)

    with open(f"images_for_every_block/pic_{manzana[i]}.jpg", 'wb') as f:
        _ =f.write(r.content)



import requests

import pandas as pd

GOOGLE_API_KEY = "..."

data = pd.read_excel('2022-05-04-225400_pi_00_port_inmob_2022-05-04-225216.xlsx', engine='openpyxl')

adds = data['address'].tolist()

adds2 = [i.replace("\n","") for i in adds]
locs = []

def extract_lat_long_via_address(address_or_zipcode):
    lat, lng = None, None
    api_key = GOOGLE_API_KEY
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?address={address_or_zipcode}&key={api_key}"
    # see how our endpoint includes our API key? Yes this is yet another reason to restrict the key
    r = requests.get(endpoint)
    if r.status_code not in range(200, 299):
        return None, None
    try:
        '''
        This try block incase any of our inputs are invalid. This is done instead
        of actually writing out handlers for all kinds of responses.
        '''
        results = r.json()['results'][0]
        lat = results['geometry']['location']['lat']
        lng = results['geometry']['location']['lng']
    except:
        pass
    return lat, lng


for i in adds2:
    locs.append(extract_lat_long_via_address(i))

data['geocode'] = locs

data.to_excel("lat_lon_nuevos_02.xlsx", index = False)


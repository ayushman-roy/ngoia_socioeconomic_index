import os
import pandas as pd
import requests
from dotenv import load_dotenv
import time
from geopy.distance import geodesic

load_dotenv()
API_KEY = os.getenv("OPENCAGE_API_KEY")

# add distance from T1 cities 
cities = ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]

absolute_path = os.path.dirname(__file__)
df_districts = pd.read_csv(os.path.join(absolute_path,'data/MASTERSHEET.csv'))

def get_geocode(address, api_key):
    """Get latitude and longitude for a given address using OpenCage Data API."""
    url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()['results']
        if results:
            geometry = results[0]['geometry']
            return geometry['lat'], geometry['lng']
    return None, None

# cache coordinates 
city_coords = {}
for city in cities:
    lat, lon = get_geocode(city, API_KEY)
    if lat is not None and lon is not None:
        city_coords[city] = (lat, lon)
    else:
        print(f"Coordinates not found for {city}")
    # Pause to respect API rate limits
    time.sleep(1)

results_list = []

for idx, row in df_districts.iterrows():
    address = f"{row['district']}, {row['state']}"
    d_lat, d_lon = get_geocode(address, API_KEY)
    
    if d_lat is None or d_lon is None:
        print(f"Skipping {address} due to missing coordinates")
        continue
    
    district_coords = (d_lat, d_lon)
    print(district_coords)
    
    min_distance = float('inf')
    closest_city = None

    # distance from the district to each target city
    for city, coords in city_coords.items():
        city_distance = geodesic(district_coords, coords).kilometers
        if city_distance < min_distance:
            min_distance = city_distance
            closest_city = city

    results_list.append({
        'district': row['district'],
        'state': row['state'],
        'city': closest_city,
        'distance': min_distance
    })

    # pause between API calls
    time.sleep(1)
    
df_results = pd.DataFrame(results_list)

df_results.to_csv((os.path.join(absolute_path, "distances_from_t1_cities.csv")), index=False)


# merging back to master dataset
master_df = pd.read_csv(os.path.join(absolute_path,'data/MASTERSHEET.csv'))
distances_df = pd.read_csv(os.path.join(absolute_path,'data/distances_from_t1_cities.csv'))

distances_df = distances_df[['district', 'state', 'distance']]

merged_df = pd.merge(master_df, distances_df, on=['state', 'district'], how='left')

merged_df.to_csv((os.path.join(absolute_path,'data/MASTERSHEET.csv')), index=False)

print("Merged CSV created successfully as 'MASTERSHEET.csv'.")
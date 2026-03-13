import pandas as pd
import requests
from datetime import datetime
from io import StringIO


#50149 is Edmonton International Airport
#50430 is Calgary International Airport
stations = {
    "Edmonton": 50149,
    "Calgary": 50430
}

start_year = 2021
end_year = 2024

url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

datasets = []

for city, station_id in stations.items():
    print(f"Fetching data for {city}")
    
    for year in range(start_year, end_year + 1):
        for month in range(1,13):
            params = {
                "format": "csv",
                "stationID": station_id,
                "Year": year,
                "Month": month,
                "Day": 1,
                "timeframe": 1, #hourly
                }
        
        try:
            r = requests.get(url, params=params)
            if "Station Name" not in r.text:
                print(f"No data for {city} for {year}-{month}")
                continue
            
            df = pd.read_csv(StringIO(r.text))
            df["City"] = city
            datasets.append(df)
            print(f"{city} for {year}-{month:02d}months downloaded")
        
        except Exception as e:
            print(f"Error {city} for {year}-{month:02d}: {e}")
            
if datasets:
    final_df = pd.concat(datasets, ignore_index =True)
    final_df.to_csv("historical_weather_datapull.csv", index= False)
    print("\nSaved to historical_weather_datapull.csv")
else:
    print("No data downloaded.")
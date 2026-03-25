#NOT WORKING PROPERLY
#TO-DO:  Fix so datapull from env Canada works properly

from env_canada import ECWeather
import pandas as pd
from pathlib import Path

# Edmonton airport (station 30122, ~YEG)
weather = ECWeather(station=30122)
df_edm = weather.weather(start="2020-01-01", end="2024-12-31")

# Save hourly temp, wind, precip
df_weather = df_edm[['date_time_pst', 'temp']].rename(columns={
    'date_time_pst': 'datetime', 
    'temp': 'temperature_C'
}).dropna()

df_weather.to_csv("edmonton_weather.csv", index=False)
print(f"Weather data: {len(df_weather)} rows")
print(df_weather.head())

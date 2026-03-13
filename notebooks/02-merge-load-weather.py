import pandas as pd
from pathlib import Path

data_dir = Path("data")
raw_dir = Path("data/raw")
proc_dir = Path("data/processed")

# Load weather data file
df_weather = pd.read_csv(raw_dir / "historical_weather_datapull.csv", encoding="latin1", low_memory=False)

#Change weather data's Date column to DATETIME, floor to hour for merging to load dataS
df_weather["DATETIME"] = (pd.to_datetime(df_weather["Date/Time (LST)"], errors="coerce").dt.floor("h"))

#print(df_weather.columns)
#print(df_weather.columns.tolist())
#Rename column names (UTF-8) for clarity and consistency, check column names in python compiler with print statement above
df_weather = df_weather.rename(columns={
    "Temp (Â°C)": "temp_C",
    "Rel Hum (%)": "rel_hum_pct",
    "Wind Spd (km/h)": "wind_kmh",
    "City": "city"
})

#Keep relevant columns with features of interested for project ie) Datetime, city, temp_c, relative humidity, wind speed
df_weather = df_weather[[
    "DATETIME",
    "city",
    "temp_C",
    "rel_hum_pct",
    "wind_kmh",
]]

#Aggregate for duplicate check
df_weather = df_weather.groupby(["DATETIME","city"], as_index=False).agg({"temp_C":"mean",
        "rel_hum_pct":"mean",
        "wind_kmh":"mean"
    })
    
#Load in the AESO Alberta load data
df_load = pd.read_csv(proc_dir / "aeso_load_clean.csv", parse_dates=["DATETIME"])
df_load["DATETIME"] = pd.to_datetime(df_load["DATETIME"]).dt.floor("h")
df_load = df_load[df_load["DATETIME"] >= '2011-12-01']
df_load = df_load.sort_values("DATETIME").reset_index(drop=True)   

#Correct timestamp of weather/load discrepancy using merge_asof
df_weather = df_weather.sort_values("DATETIME")

#Split data by city
df_edm = df_weather[df_weather["city"] == "Edmonton"].copy()
df_cal = df_weather[df_weather["city"] == "Calgary"].copy()

# Rename columns to be city-specific
df_edm = df_edm.rename(columns={
    "temp_C": "temp_edm_C",
    "rel_hum_pct": "rel_hum_edm_pct",
    "wind_kmh": "wind_edm_kmh"
}).drop(columns=["city"])

df_cal = df_cal.rename(columns={
    "temp_C": "temp_cgy_C",
    "rel_hum_pct": "rel_hum_cgy_pct",
    "wind_kmh": "wind_cgy_kmh"
}).drop(columns=["city"])

#Merge Data sets
df_merged = pd.merge_asof(df_load, df_edm, on="DATETIME", direction = "backward")
df_merged = pd.merge_asof(df_merged, df_cal[["DATETIME","temp_cgy_C","rel_hum_cgy_pct","wind_cgy_kmh"]], on="DATETIME", direction = "backward")

#Add some time features for exploration
df_merged["hour"] = df_merged["DATETIME"].dt.hour
df_merged["day_of_week"] = df_merged["DATETIME"].dt.dayofweek
df_merged["month"] = df_merged["DATETIME"].dt.month
df_merged["is_weekend"] = df_merged["day_of_week"].isin([5,6]).astype(int)

#Drop Na Values from weather pull
df_merged = df_merged.dropna(subset=["temp_edm_C", "temp_cgy_C"])

#Save output
proc_dir.mkdir(parents=True, exist_ok=True)
df_merged.to_csv(proc_dir / "modelling_dataset.csv", index=False)

#Diagnostic Checks for dataframe integrity
print(df_merged.head())
print(df_merged.isna().mean())
print(df_merged.shape)
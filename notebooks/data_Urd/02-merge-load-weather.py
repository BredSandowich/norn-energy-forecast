import pandas as pd
from pathlib import Path
import yaml

##If not using yaml
#data_dir = Path("data")
#raw_dir = Path("data/raw")
#proc_dir = Path("data/processed")

#Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

raw_dir = Path(config["paths"]["raw_data"])
proc_dir = Path(config["paths"]["processed_data"])
proc_dir.mkdir(parents=True, exist_ok=True)

# Load weather data file
df_weather = pd.read_csv(raw_dir / "historical_weather_datapull.csv", encoding="latin1", low_memory=False)

#Change/standardize weather data's Date column to DATETIME
df_weather["Datetime"] = pd.to_datetime(df_weather["Date/Time (LST)"], errors="coerce")
#print(df_weather.columns)
#print(df_weather.columns.tolist())

#Rename column names (UTF-8) for clarity and consistency, check column names in python compiler with print statement above
df_weather = df_weather.rename(columns=lambda x: x.strip())
df_weather = df_weather.rename(columns={
    "Temp (Â°C)": "temp_C",
    "Rel Hum (%)": "rel_hum_pct",
    "Wind Spd (km/h)": "wind_kmh",
    "City": "city"
})

#Keep relevant columns with features of interested for project ie) Datetime, city, temp_c, relative humidity, wind speed
df_weather = df_weather[[
    "Datetime",
    "city",
    "temp_C",
    "rel_hum_pct",
    "wind_kmh",
]]

#Aggregate for duplicate check
df_weather = df_weather.groupby(["Datetime","city"], as_index=False).agg({
    "temp_C":"mean",
    "rel_hum_pct":"mean",
    "wind_kmh":"mean"
    })
    
#Load in the AESO Alberta load data
df_load = pd.read_csv(proc_dir / "aeso_load_clean.csv", parse_dates=["Datetime"])
df_load["Datetime"] = pd.to_datetime(df_load["Datetime"])
df_load = df_load[df_load["Datetime"] >= df_weather["Datetime"].min()]
df_load = df_load.sort_values("Datetime").reset_index(drop=True)   

#Correct timestamp of weather/load discrepancy using merge_asof
df_weather = df_weather.sort_values("Datetime")

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

#Adjust for difference in weather data start dates for each city
weather_start = max(df_edm["Datetime"].min(), df_cal["Datetime"].min())
df_load = df_load[df_load["Datetime"] >= weather_start].copy()

#Merge Data sets
df_merged = pd.merge_asof(df_load.sort_values("Datetime"), df_edm.sort_values("Datetime"), on="Datetime", direction = "backward")
df_merged = pd.merge_asof(df_merged, df_cal[["Datetime","temp_cgy_C","rel_hum_cgy_pct","wind_cgy_kmh"]], on="Datetime", direction = "backward")

#Add some time features for exploration
df_merged["hour"] = df_merged["Datetime"].dt.hour
df_merged["day_of_week"] = df_merged["Datetime"].dt.dayofweek
df_merged["month"] = df_merged["Datetime"].dt.month
df_merged["is_weekend"] = df_merged["day_of_week"].isin([5,6]).astype(int)

##Drop Na Values from weather pull
#df_merged = df_merged.dropna(subset=["temp_edm_C", "temp_cgy_C"])
#print(df_merged.isna().mean())

#Add a flag for filling in missing weather data
# Create flags for Edmonton and Calgary weather
df_merged["edm_weather_missing"] = df_merged[["temp_edm_C", "rel_hum_edm_pct", "wind_edm_kmh"]].isna().any(axis=1).astype(int)
df_merged["cgy_weather_missing"] = df_merged[["temp_cgy_C", "rel_hum_cgy_pct", "wind_cgy_kmh"]].isna().any(axis=1).astype(int)

#Interpolate the gaps in temperature data instead of dropping them for future feature modelling in time series. Will tighten up with stricter interpolation in final pipeline
df_merged["temp_edm_C"] = df_merged["temp_edm_C"].interpolate(limit=3)
df_merged["temp_cgy_C"] = df_merged["temp_cgy_C"].interpolate(limit=3)

df_merged["rel_hum_edm_pct"] = df_merged["rel_hum_edm_pct"].interpolate(limit=3)
df_merged["rel_hum_cgy_pct"] = df_merged["rel_hum_cgy_pct"].interpolate(limit=3)

df_merged["wind_edm_kmh"] = df_merged["wind_edm_kmh"].interpolate(limit=3)
df_merged["wind_cgy_kmh"] = df_merged["wind_cgy_kmh"].interpolate(limit=3)

#print(df_merged.isna().mean())

#Forward fill some more data gaps in weather information
df_merged["temp_edm_C"] = df_merged["temp_edm_C"].ffill()
df_merged["temp_cgy_C"] = df_merged["temp_cgy_C"].ffill()

df_merged["rel_hum_edm_pct"] = df_merged["rel_hum_edm_pct"].ffill()
df_merged["rel_hum_cgy_pct"] = df_merged["rel_hum_cgy_pct"].ffill()

df_merged["wind_edm_kmh"] = df_merged["wind_edm_kmh"].ffill()
df_merged["wind_cgy_kmh"] = df_merged["wind_cgy_kmh"].ffill()
print(df_merged.isna().mean())

#Save output
##Without yaml
#proc_dir.mkdir(parents=True, exist_ok=True)
#df_merged.to_csv(proc_dir / "modelling_dataset.csv", index=False)

output_path = proc_dir / "modelling_dataset.csv"
df_merged.to_csv(output_path, index=False)

#Diagnostic Checks for dataframe integrity
#print(df_merged.head())
#print(df_merged.isna().mean())
#print(df_merged.shape)

print(f"Saved merged dataset to {output_path}")
print(f"Shape: {df_merged.shape}")

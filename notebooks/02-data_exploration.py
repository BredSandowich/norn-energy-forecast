import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set working directory to project root
os.chdir('/workspaces/norn-energy-forecast')

# Load the dataset
df = pd.read_csv('data/processed/modelling_dataset.csv')

# Parse datetime
df['DATETIME'] = pd.to_datetime(df['DATETIME'])

#Extract Year for yearly trend analysis
df["year"] = df["DATETIME"].dt.year

# Drop unnecessary column, already in datetime column created
df = df.drop(columns=['HOUR_ENDING_RAW'])

# Rename columns for consistency
df = df.rename(columns={
    'EDMONTON': 'load_edm_mw',
    'CALGARY': 'load_cgy_mw',
    'rel_hum_ed_pct': 'rel_hum_edm_pct',
})

# Create separate dataframes for each city
edmonton_df = df[['DATETIME', 'load_edm_mw', 'temp_edm_C', 'rel_hum_edm_pct', 'wind_edm_kmh', 'hour', 'day_of_week', 'month', 'is_weekend','year']].copy()
calgary_df = df[['DATETIME', 'load_cgy_mw', 'temp_cgy_C', 'rel_hum_cgy_pct', 'wind_cgy_kmh', 'hour', 'day_of_week', 'month', 'is_weekend','year']].copy()

print("=== EDMONTON DATA ===")
print("Shape:", edmonton_df.shape)
print("\nRanges:")
print(edmonton_df.describe())
print("\nMissing values:")
print(edmonton_df.isnull().sum())

print("\n=== CALGARY DATA ===")
print("Shape:", calgary_df.shape)
print("\nRanges:")
print(calgary_df.describe())
print("\nMissing values:")
print(calgary_df.isnull().sum())

## EDA Plots ##

# Load vs Temperature plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Edmonton load to temperature plot
axes[0].scatter(edmonton_df['temp_edm_C'], edmonton_df['load_edm_mw'], alpha=0.5)
axes[0].set_title('Edmonton: Load (MW) vs Temperature (°C)')
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Load (MW)')

# Calgary lod to temperature plot
axes[1].scatter(calgary_df['temp_cgy_C'], calgary_df['load_cgy_mw'], alpha=0.5)
axes[1].set_title('Calgary: Load (MW) vs Temperature (°C)')
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_ylabel('Load (MW)')

plt.tight_layout()
plt.savefig('notebooks/load_vs_temp.png')
print("\nLoad vs Temperature plots saved to notebooks/load_vs_temp.png")

# Average load by hour
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

edm_hourly = edmonton_df.groupby('hour')['load_edm_mw'].mean()
axes[0].plot(edm_hourly.index, edm_hourly.values, marker='o')
axes[0].set_title('Edmonton: Average Load by Hour of Day')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Average Load (MW)')
axes[0].grid(True)

cgy_hourly = calgary_df.groupby('hour')['load_cgy_mw'].mean()
axes[1].plot(cgy_hourly.index, cgy_hourly.values, marker='o')
axes[1].set_title('Calgary: Average Load by Hour of Day')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Average Load (MW)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('notebooks/avg_load_by_hour.png')
print("Average load by hour plots saved to notebooks/avg_load_by_hour.png")

# Average load by weekday/weekend
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

edm_weekend = edmonton_df.groupby('is_weekend')['load_edm_mw'].mean()
axes[0].bar(['Weekday', 'Weekend'], edm_weekend.values)
axes[0].set_title('Edmonton: Average Load by Weekday/Weekend')
axes[0].set_ylabel('Average Load (MW)')

cgy_weekend = calgary_df.groupby('is_weekend')['load_cgy_mw'].mean()
axes[1].bar(['Weekday', 'Weekend'], cgy_weekend.values)
axes[1].set_title('Calgary: Average Load by Weekday/Weekend')
axes[1].set_ylabel('Average Load (MW)')

plt.tight_layout()
plt.savefig('notebooks/avg_load_weekday_weekend.png')
print("Average load by weekday/weekend plots saved to notebooks/avg_load_weekday_weekend.png")

# Average load by month
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

edm_monthly = edmonton_df.groupby('month')['load_edm_mw'].mean()
axes[0].plot(edm_monthly.index, edm_monthly.values, marker='o')
axes[0].set_title('Edmonton: Average Load by Month')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Average Load (MW)')
axes[0].grid(True)

cgy_monthly = calgary_df.groupby('month')['load_cgy_mw'].mean()
axes[1].plot(cgy_monthly.index, cgy_monthly.values, marker='o')
axes[1].set_title('Calgary: Average Load by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Average Load (MW)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('notebooks/avg_load_by_month.png')
print("Average load by month plots saved to notebooks/avg_load_by_month.png")

# Average load by year
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

edm_yearly = edmonton_df.groupby('year')['load_edm_mw'].mean()
axes[0].plot(edm_yearly.index, edm_yearly.values, marker='o')
axes[0].set_title('Edmonton: Average Load by Month')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Average Load (MW)')
axes[0].grid(True)

cgy_yearly = calgary_df.groupby('year')['load_cgy_mw'].mean()
axes[1].plot(cgy_yearly.index, cgy_yearly.values, marker='o')
axes[1].set_title('Calgary: Average Load by Yearly')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Load (MW)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('notebooks/avg_load_by_yearly.png')
print("Average load by hour plots saved to notebooks/avg_load_by_yearly.png")

print("\nAnalysis complete. Plots saved in notebooks/ directory.")
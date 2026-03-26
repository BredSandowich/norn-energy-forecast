import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import calendar

'''
# Set working directory to project root
os.chdir('/workspaces/norn-energy-forecast')
# Load the dataset
df = pd.read_csv('data/processed/modelling_dataset.csv')
'''
#Changed to use yaml, uncomment above if not using yaml file
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

proc_dir = Path(config["paths"]["processed_data"])
eda_dir = Path(config["paths"]["eda"])

#Read
df = pd.read_csv(proc_dir / "modelling_dataset.csv", encoding="latin1", low_memory=False)

##Use pathlib to navigate and put reports into reports/ folder of repo
#report_dir = Path("reports/figures")
report_dir = Path(config["paths"]["reports"])
report_dir.mkdir(parents=True, exist_ok=True)
#Function for saving plots to reports folder/
def save_plot(filename):
    plt.tight_layout()
    path = report_dir / filename
    plt.savefig(path, bbox_inches="tight")
    print(f"saved plot: {path}")
    plt.close()
    
# Parse datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

#Extract Year for yearly trend analysis
df["year"] = df["Datetime"].dt.year

# Rename columns for consistency
df = df.rename(columns={
    'EDMONTON': 'load_edm_mw',
    'CALGARY': 'load_cgy_mw',
    'rel_hum_ed_pct': 'rel_hum_edm_pct',
})

# Create separate dataframes for each city
edmonton_df = df[['Datetime', 'load_edm_mw', 'temp_edm_C', 'rel_hum_edm_pct', 'wind_edm_kmh', 'hour', 'day_of_week', 'month', 'is_weekend','year']].copy()
calgary_df = df[['Datetime', 'load_cgy_mw', 'temp_cgy_C', 'rel_hum_cgy_pct', 'wind_cgy_kmh', 'hour', 'day_of_week', 'month', 'is_weekend','year']].copy()

#Basic Stats for each dataframe
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

# Calgary load to temperature plot
axes[1].scatter(calgary_df['temp_cgy_C'], calgary_df['load_cgy_mw'], alpha=0.5)
axes[1].set_title('Calgary: Load (MW) vs Temperature (°C)')
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_ylabel('Load (MW)')

'''
plt.tight_layout()
plt.savefig('notebooks/load_vs_temp.png')
print("\nLoad vs Temperature plots saved to reports/load_vs_temp.png")
'''
save_plot("load_vs_temp.png")

#Second Edmonton Scatterplot with correlation
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"]=(12,6)
sns.scatterplot(x="temp_edm_C", y="load_edm_mw", data=edmonton_df, alpha=0.5)
plt.title("Edmonton Load vs Temperature")
#plt.savefig('notebooks/load_vs_temp_edm2.png')
print(edmonton_df["load_edm_mw"].corr(edmonton_df['temp_edm_C']))
save_plot("ed_load_vs_temp.png")

#Second Calgary Scatterplot with correlation
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"]=(12,6)
sns.scatterplot(x="temp_cgy_C", y="load_cgy_mw", data=calgary_df, alpha=0.5)
plt.title("Calgary Load vs Temperature")
#plt.savefig('notebooks/load_vs_temp_edm2.png')
print(calgary_df["load_cgy_mw"].corr(calgary_df['temp_cgy_C']))
save_plot("cgy_load_vs_temp.png")


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

#plt.tight_layout()
#plt.savefig('notebooks/avg_load_by_hour.png')
save_plot("avg_load_by_hour.png")
print("Average load by hour plots saved to reports/avg_load_by_hour.png")

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

#plt.tight_layout()
#plt.savefig('notebooks/avg_load_weekday_weekend.png')
save_plot("avg_load_weekday_weekend.png")
print("Average load by weekday/weekend plots saved to reports/avg_load_weekday_weekend.png")


# Average load by month
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
edm_monthly = edmonton_df.groupby('month')['load_edm_mw'].mean()
axes[0].plot(edm_monthly.index, edm_monthly.values, marker='o')
axes[0].set_title('Edmonton: Average Load by Month')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Average Load (MW)')
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels([calendar.month_abbr[m] for m in edm_monthly.index])
axes[0].grid(True)

cgy_monthly = calgary_df.groupby('month')['load_cgy_mw'].mean()
axes[1].plot(cgy_monthly.index, cgy_monthly.values, marker='o')
axes[1].set_title('Calgary: Average Load by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Average Load (MW)')
axes[1].set_xticks(range(1,13))
axes[1].set_xticklabels([calendar.month_abbr[m] for m in cgy_monthly.index])
axes[1].grid(True)

#plt.tight_layout()
#plt.savefig('notebooks/avg_load_by_month.png')
save_plot("avg_load_by_month.png")
print("Average load by month plots saved to reports/avg_load_by_month.png")


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
#plt.savefig('notebooks/avg_load_by_yearly.png')
save_plot("avg_load_by_yearly.png")
print("Average load by hour plots saved to reports/avg_load_by_yearly.png")


# Average load by day of week
day_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
edm_doweek = edmonton_df.groupby('day_of_week')['load_edm_mw'].mean()
axes[0].plot(edm_doweek.index, edm_doweek.values, marker='o')
axes[0].set_title('Edmonton: Average Load by Day of Week')
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Average Load (MW)')
axes[0].set_xticks(edm_doweek.index)
axes[0].set_xticklabels([day_of_week[d] for d in edm_doweek.index])
axes[0].grid(True)

cgy_doweek = calgary_df.groupby('day_of_week')['load_cgy_mw'].mean()
axes[1].plot(cgy_doweek.index, cgy_doweek.values, marker='o')
axes[1].set_title('Calgary: Average Load by Day of Week')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Average Load (MW)')
axes[1].set_xticks(cgy_doweek.index)
axes[1].set_xticklabels([day_of_week[d] for d in cgy_doweek.index])
axes[1].grid(True)

#plt.tight_layout()
#plt.savefig('notebooks/avg_load_by_doweek.png')
save_plot("avg_load_by_doweek.png")
print("Average load by day of week plots saved to reports/avg_load_by_doweek.png")

'''
#Correlation Heatmap Edmonton
plt.figure(figsize=(10, 6))
sns.heatmap(edmonton_df[["load_edm_mw", "temp_edm_C","rel_hum_edm_pct", "wind_edm_kmh"]].corr(), annot=True, cmap='coolwarm')
plt.title("Edmonton: Correlation Heatmap")
#plt.savefig('notebooks/heatmap_edmonton.png')
save_plot("heatmap_edmonton.png")
print("Correlation heatmap for Edmonton saved to reports/heatmap_edmonton.png")

#Correlation Heatmap Calgary
plt.figure(figsize=(10, 6))
sns.heatmap(calgary_df[["load_cgy_mw", "temp_cgy_C","rel_hum_cgy_pct", "wind_cgy_kmh"]].corr(), annot=True, cmap='coolwarm')
plt.title("Calgary: Correlation Heatmap")
#plt.savefig('notebooks/heatmap_edmonton.png')
save_plot("heatmap_calgary.png")
print("Correlation heatmap for Calgary saved to reports/heatmap_calgary.png")


#Time-series plot of load
plt.figure(figsize=(12, 6))
plt.plot(df['Datetime'], df['load_edm_mw'], label='Edmonton Load', alpha=0.7)
plt.plot(df['Datetime'], df['load_cgy_mw'], label='Calgary Load', alpha=0.7)
plt.title("Time Series of Load (MW)")
plt.xlabel("Datetime")
plt.ylabel("Load (MW)")
plt.legend()
save_plot("time_series_load.png")


#Edmonton Temperature vs load Scatter plot w/ trendline
edm_temp_load_smoothed = edmonton_df[['temp_edm_C', 'load_edm_mw']].rolling(100).mean()

sns.scatterplot(x="temp_edm_C", y="load_edm_mw", data=edmonton_df, alpha=0.3)
sns.lineplot(x="temp_edm_C", y="load_edm_mw", data=edm_temp_load_smoothed, color='red')
plt.title("Edmonton Load vs Temperature")
save_plot("temp_vs_load_edm.png")

#Calgary Temperature vs load Scatter plot w/ trendline
cgy_temp_load_smoothed = calgary_df[['temp_cgy_C', 'load_cgy_mw']].rolling(100).mean()

sns.scatterplot(x="temp_cgy_C", y="load_cgy_mw", data=calgary_df, alpha=0.3)
sns.lineplot(x="temp_cgy_C", y="load_cgy_mw", data=cgy_temp_load_smoothed, color='red')
plt.title("Calgary Load vs Temperature")
save_plot("temp_vs_load_cgy.png")
'''

#Feature engineering (commonly used feature such as heating degree days and cooling degree days)
HDD_base = 18.0
CDD_base = 22.0
edmonton_df["HDD"] = (HDD_base - edmonton_df["temp_edm_C"]).clip(lower=0)
edmonton_df["CDD"] = (edmonton_df["temp_edm_C"] - CDD_base).clip(lower=0)   
calgary_df["HDD"] = (HDD_base - calgary_df["temp_cgy_C"]).clip(lower=0)
calgary_df["CDD"] = (calgary_df["temp_cgy_C"] - CDD_base).clip(lower=0)   

#Visualization of features HDD/CDD
sns.scatterplot(x="HDD", y="load_edm_mw", data=edmonton_df,alpha=0.3)
plt.title("Edmonton Load vs Heating Degree Days")
plt.savefig('reports/edm_load_vs_HDD.png')
print("Edmonton Load vs Heating Degree Days plot saved to reports/edm_load_vs_HDD.png")      

sns.scatterplot(x="CDD", y="load_edm_mw", data=edmonton_df, alpha=0.3,color='orange')
plt.title("Edmonton Load vs Cooling Degree Days")
plt.savefig('reports/edm_load_vs_CDD.png')
print("Edmonton Load vs Cooling Degree Days plot saved to reports/edm_load_vs_CDD.png")

sns.scatterplot(x="HDD", y="load_cgy_mw", data=calgary_df,alpha=0.3)
plt.title("Calgary Load vs Heating Degree Days")
plt.savefig('reports/cgy_load_vs_HDD.png')
print("Calgary Load vs Heating Degree Days plot saved to reports/cgy_load_vs_HDD.png")      

sns.scatterplot(x="CDD", y="load_cgy_mw", data=calgary_df, alpha=0.3,color='orange')
plt.title("Calgary Load vs Cooling Degree Days")
plt.savefig('reports/cgy_load_vs_CDD.png')
print("Edmonton Load vs Cooling Degree Days plot saved to reports/cgy_load_vs_CDD.png")

#Visualization of load and temp for 2024
latest_year = df["year"].max()
df_latest_year = df[df["year"] == latest_year].copy()

fig, ax1 = plt.subplots(figsize=(14,6))
#Load shown on left axis
ax1.plot(df_latest_year["Datetime"], df_latest_year["load_edm_mw"], color= "blue", label = "Load (MW)")
ax1.set_xlabel("Datetime")
ax1.set_ylabel("Load (MW)", color="Blue")
ax1.tick_params(axis='y', labelcolor="blue")

#Temp shown on right axis
ax2 = ax1.twinx()
ax2.plot(df_latest_year["Datetime"], df_latest_year["temp_edm_C"], color= "red", label = "Temp C")
ax2.set_ylabel("Temp (°C)", color="Red")
ax2.tick_params(axis='y', labelcolor="red")

plt.title(f"Edmonton Load vs Temperature ({latest_year})")

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

save_plot("load_vs_temp_timeseries.png")


#Highlight the HDD and CDD for interest sake of highlighting extreme temp. events
edmonton_df["HDD"] = (HDD_base - df_latest_year["temp_edm_C"]).clip(lower=0)
edmonton_df["CDD"] = (df_latest_year["temp_edm_C"] - CDD_base).clip(lower=0)  

# Define extreme thresholds (tune as needed)
extreme_cold = df_latest_year["temp_edm_C"] < -20 #df_latest_year["temp_edm_C"].quantile(0.9)
extreme_hot = df_latest_year["temp_edm_C"] > 25 #df_latest_year["temp_edm_C"].quantile(0.9)

fig, ax1 = plt.subplots(figsize=(14, 6))

# Load
ax1.plot(df_latest_year['Datetime'], df_latest_year['load_edm_mw'], color='blue', label='Load (MW)')

# Highlight extreme points
ax1.scatter(df_latest_year.loc[extreme_cold, 'Datetime'],
            df_latest_year.loc[extreme_cold, 'load_edm_mw'],
            color='cyan', label='Extreme Cold', s=10)

ax1.scatter(df_latest_year.loc[extreme_hot, 'Datetime'],
            df_latest_year.loc[extreme_hot, 'load_edm_mw'],
            color='orange', label='Extreme Heat', s=10)

ax1.set_ylabel("Load (MW)")

# Temperature axis
ax2 = ax1.twinx()
ax2.plot(df_latest_year['Datetime'], df_latest_year['temp_edm_C'], color='red', alpha=0.5)

plt.title(f"Extreme Temperature Events vs Load ({latest_year})")
plt.legend()

save_plot("extreme_temp_vs_load.png")



#Mean load printouts
print("\nMean Load (MW)by hour:")
print(df.groupby('hour')[['load_edm_mw', 'load_cgy_mw']].mean())   

print("\nMean Load (MW) by month:")
print(df.groupby('month')[['load_edm_mw', 'load_cgy_mw']].mean())

print(f"\nEdmonton correlation with temperature: {edmonton_df[["load_edm_mw","temp_edm_C"]].corr()}")
print(f"\nCalgary correlation with temperature: {calgary_df[["load_cgy_mw","temp_cgy_C"]].corr()}")

print("\nPeak Load by Month:")
print(df.groupby('month')[['load_edm_mw', 'load_cgy_mw']].max())

print("\nAnalysis complete. Plots saved in:", report_dir)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

#Create common features for both cities

#Lag features to capture temporal dependencies
df["lag_1"] = df["load_edm_mw"].shift(1)
df["lag_24"] = df["load_edm_mw"].shift(24) #yesterday same hour
df["lag_168"] = df["load_edm_mw"].shift(168) #Weekly lag

#Rolling mean features to capture trends
df["rolling_mean_24"] = df["load_edm_mw"].rolling(window=24).mean().shift(1)
df["rolling_mean_168"] = df["load_edm_mw"].rolling(window=168).mean().shift(1)
df["rolling_std_24"] = df["load_edm_mw"].rolling(window=24).std().shift(1)
df["rolling_std_168"] = df["load_edm_mw"].rolling(window=168).std().shift(1)

#Lag features to capture temporal dependencies
df["lag_1"] = df["load_cgy_mw"].shift(1)
df["lag_24"] = df["load_cgy_mw"].shift(24) #yesterday same hour
df["lag_168"] = df["load_cgy_mw"].shift(168) #Weekly lag

#Rolling mean features to capture trends
df["rolling_mean_24"] = df["load_cgy_mw"].rolling(window=24).mean().shift(1)
df["rolling_mean_168"] = df["load_cgy_mw"].rolling(window=168).mean().shift(1)
df["rolling_std_24"] = df["load_cgy_mw"].rolling(window=24).std().shift(1)
df["rolling_std_168"] = df["load_cgy_mw"].rolling(window=168).std().shift(1)


# Drop rows with NaN values introduced by lag and rolling features, keeping some missing weather features as to not lose data
df = df.dropna(subset=[
    "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168", "rolling_std_24", "rolling_std_168"
    ])

#Setup forecast evaluation metrics
def forecast_evaluation(y_true, y_pred):
    error = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(error / y_true)) * 100
    return error, mae, rmse, mape

# Create separate dataframes for each city
edmonton_df = df[['DATETIME', 'load_edm_mw', 'temp_edm_C', 'rel_hum_edm_pct', 'wind_edm_kmh', 'hour', 'day_of_week', 'month', 'is_weekend','year', 'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168', 'rolling_std_24', 'rolling_std_168']].copy()
calgary_df = df[['DATETIME', 'load_cgy_mw', 'temp_cgy_C', 'rel_hum_cgy_pct', 'wind_cgy_kmh', 'hour', 'day_of_week', 'month', 'is_weekend','year', 'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168', 'rolling_std_24', 'rolling_std_168']].copy()


#Train for 7 day forecast
horizon = 24*7
train_df = edmonton_df[:-horizon]
test_df = edmonton_df[-horizon:]

features = ['temp_edm_C', 'rel_hum_edm_pct', 'wind_edm_kmh', 'hour', 'day_of_week', 'month', 'is_weekend', 'year', 'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168', 'rolling_std_24', 'rolling_std_168']
target = 'load_edm_mw'

x_train = train_df[features]
y_train = train_df[target]
x_test = test_df[features]
y_test = test_df[target]

#Train a Simple Model - Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(x_train, y_train)
rf_forecast = rf_model.predict(x_test)

# Evaluate Random Forest forecast
rf_results = forecast_evaluation(y_test, rf_forecast)
rf_mae, rf_rmse, rf_mape = rf_results[1], rf_results[2], rf_results[3]
print(f"Random Forest - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, MAPE: {rf_mape:.2f}%")

#Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(test_df['DATETIME'], y_test, label='Actual Load', marker='o')
plt.plot(test_df['DATETIME'], rf_forecast, label=f'Random Forest Forecast (MAPE: {rf_mape:.2f}%)', marker='x')
plt.title('Random Forest Forecast vs Actual Load')
plt.xlabel('DateTime')
plt.ylabel('Load (MW)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/rf_forecast.png')
print("\nRandom Forest forecast plot saved to notebooks/rf_forecast.png")

#Train a Simple Model - Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_forecast = lr_model.predict(x_test)
# Evaluate Linear Regression forecast
lr_results = forecast_evaluation(y_test, lr_forecast)
lr_mae, lr_rmse, lr_mape = lr_results[1], lr_results[2], lr_results[3]
print(f"Linear Regression - MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}, MAPE: {lr_mape:.2f}%")

#Plot actual vs forecast Linear Regression
plt.figure(figsize=(12, 6))
plt.plot(test_df['DATETIME'], y_test, label='Actual Load', marker='o')
plt.plot(test_df['DATETIME'], lr_forecast, label=f'Linear Regression Forecast (MAPE: {lr_mape:.2f}%)', marker='x')
plt.title('Linear Regression Forecast vs Actual Load')
plt.xlabel('DateTime')
plt.ylabel('Load (MW)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/lr_forecast.png')
print("\nLinear Regression forecast plot saved to notebooks/lr_forecast.png")

#Ensemble Seasonal Naive Forecast and Random Forest
ensemble_forecast = (rf_forecast + test_df['lag_24'].values) / 2
ensemble_results = forecast_evaluation(y_test, ensemble_forecast)
ensemble_mae, ensemble_rmse, ensemble_mape = ensemble_results[1], ensemble_results[2], ensemble_results[3]
print(f"Ensemble (Random Forest + Seasonal Naive) - MAE: {ensemble_mae:.2f}, RMSE: {ensemble_rmse:.2f}, MAPE: {ensemble_mape:.2f}%")
#Plot actual vs forecast Ensemble
plt.figure(figsize=(12, 6))
plt.plot(test_df['DATETIME'], y_test, label='Actual Load', marker='o')
plt.plot(test_df['DATETIME'], ensemble_forecast, label=f'Ensemble Forecast (MAPE: {ensemble_mape:.2f}%)', marker='x')
plt.title('Ensemble Forecast vs Actual Load')
plt.xlabel('DateTime')
plt.ylabel('Load (MW)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/ensemble_forecast.png')
print("\nEnsemble forecast plot saved to notebooks/ensemble_forecast.png")

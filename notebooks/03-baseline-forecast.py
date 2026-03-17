import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

#Setup forecast evaluation metrics
def forecast_evaluation(y_true, y_pred):
    error = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(error / y_true)) * 100
    return error, mae, rmse, mape

#Forecast Evaluation
#Start with 24 hour forecast
horizon = 24
y = edmonton_df['load_edm_mw']

train_naive = y[:-horizon]
test_naive = y[-horizon:]

#Baseline forecast 1: Seasonal Naive Method (rolling naive)
naive = train_naive.iloc[-24:]
naive = pd.Series(naive.values, index=test_naive.index)

#Plot Seasonal Naive forecast
plt.figure(figsize=(10, 5)) 
plt.plot(train_naive.index[-168:], train_naive.iloc[-168:], label='Train (last 7 days)')
plt.plot(test_naive.index, test_naive, label='Test')

plt.plot(naive.index, naive, label='Seasonal Naive Forecast', linestyle='--')
plt.title('Seasonal Naive Forecast vs Actual Load')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/seasonal_naive_forecast.png')

metrics_df = pd.DataFrame(columns=['Model', "Error", 'MAE', 'RMSE', 'MAPE'])
naive_results = forecast_evaluation(test_naive, naive)
naive_row = pd.DataFrame({'Model': ['Seasonal Naive'], "Error": [naive_results[0]], 'MAE': [naive_results[1]], 'RMSE': [naive_results[2]], 'MAPE': [naive_results[3]]})
metrics_df = pd.concat([metrics_df, naive_row], ignore_index=True)
print(metrics_df)

#Baseline forecast 2: Original Naive Last Point Method
naive_flat = np.repeat(train_naive.iloc[-1], horizon)
naive_flat = pd.Series(naive_flat, index=test_naive.index)

naive_flat_results = forecast_evaluation(test_naive, naive_flat)
naive_flat_row = pd.DataFrame({'Model': ['Naive (Flat)'], "Error": [naive_flat_results[0]], 'MAE': [naive_flat_results[1]], 'RMSE': [naive_flat_results[2]], 'MAPE': [naive_flat_results[3]]})
metrics_df = pd.concat([metrics_df, naive_flat_row], ignore_index=True)
print(metrics_df)


#Rolling Simple Moving Average baseline forecast (7-day seasonal)
window_days = 7
window_hours = window_days * 24
last_window = train_naive.iloc[-window_hours:]
# Reshape into days x hours
last_window_reshaped = last_window.values.reshape(window_days, 24)
# Average across days for each hour
moving_avg_forecast_values = last_window_reshaped.mean(axis=0)
moving_avg_forecast = pd.Series(moving_avg_forecast_values, index=test_naive.index)

#Plot Rolling Moving Average forecast
plt.figure(figsize=(10, 5))
plt.plot(train_naive.index[-168:], train_naive.iloc[-168:], label='Train (last 7 days)')
plt.plot(test_naive.index, test_naive, label='Test')
plt.plot(moving_avg_forecast.index, moving_avg_forecast, label='Rolling Moving Average Forecast (7-day)', linestyle='--')
plt.title('Rolling Moving Average Forecast vs Actual Load')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/rolling_moving_avg_forecast.png')

moving_avg_results = forecast_evaluation(test_naive, moving_avg_forecast)
moving_avg_row = pd.DataFrame({'Model': ['Rolling Moving Average'], "Error": [moving_avg_results[0]], 'MAE': [moving_avg_results[1]], 'RMSE': [moving_avg_results[2]], 'MAPE': [moving_avg_results[3]]})
metrics_df = pd.concat([metrics_df, moving_avg_row], ignore_index=True)
print(metrics_df)

#Baseline forecast 4: Original Simple Moving Average (flat)
window_size = 24
moving_avg_flat = train_naive.rolling(window=window_size).mean().iloc[-1]
moving_avg_flat_forecast = np.repeat(moving_avg_flat, horizon)
moving_avg_flat_forecast = pd.Series(moving_avg_flat_forecast, index=test_naive.index)

moving_avg_flat_results = forecast_evaluation(test_naive, moving_avg_flat_forecast)
moving_avg_flat_row = pd.DataFrame({'Model': ['Moving Average (Flat)'], "Error": [moving_avg_flat_results[0]], 'MAE': [moving_avg_flat_results[1]], 'RMSE': [moving_avg_flat_results[2]], 'MAPE': [moving_avg_flat_results[3]]})
metrics_df = pd.concat([metrics_df, moving_avg_flat_row], ignore_index=True)
print(metrics_df)


#Plot comparison of results
plt.figure(figsize=(10, 5))
plt.plot(test_naive.index, test_naive, label='Actual Load', marker='o')
plt.plot(naive.index, naive, label='Seasonal Naive Forecast', linestyle='--', marker='x')
plt.plot(moving_avg_forecast.index, moving_avg_forecast, label='Rolling Moving Average Forecast', linestyle='--', marker='s')
plt.plot(naive_flat.index, naive_flat, label='Naive (Flat) Forecast', linestyle='-.', marker='^')
plt.plot(moving_avg_flat_forecast.index, moving_avg_flat_forecast, label='Moving Average (Flat) Forecast', linestyle='-.', marker='v')
plt.title('Forecast Comparison')
plt.xlabel('Time')
plt.ylabel('Load (MW)')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/forecast_comparison.png')

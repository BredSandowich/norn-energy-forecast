# src/ run_pipeline_norn.py --> Runs all of the src files (Norns) in one single entry point for project Norn

import pandas as pd
from pathlib import Path
import yaml
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))

#Import pipeline src/ modules
from src.data_prep_urd import build_dataset
from src.features_verdandi import prepare_features
from src.models_skuld import seasonal_naive, rolling_moving_avg, forecast_evaluation, train_rand_forest


##Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

raw_dir = Path(config["paths"]["raw_data"])    
proc_dir = Path(config["paths"]["processed_data"])


##Load and Prep dataset
print("Building dataset...")
df = build_dataset(raw_dir, proc_dir)

## Feature engineering dataset for future machine learning (focusing on Edmonton only, can scale to change yaml file target changed for different cities depending on data pulled. Currently dataset contains Calgary as well)
#Focus model on 1 city at a time - Change yaml for different cities dependent on datapull adjustments
print("Creating features")
target = "load_edm_mw"

df = prepare_features(df, target)

##Train, test split for models
# 24 hours
horizon = 24

train = df.iloc[:-horizon]
test = df.iloc[-horizon:]

y_train = train[target]
y_test = test[target]

#Chosen baseline models after analysis
print("Running baseline models")

naive_forecast = seasonal_naive(y_train, horizon)
moving_avg_forecast = rolling_moving_avg(y_train, horizon = horizon)

naive_metrics = forecast_evaluation(y_test.values, naive_forecast.values)
moving_avg_metrics = forecast_evaluation(y_test.values, moving_avg_forecast.values)

print("\n Baseline Results:")
print("Seasonal Naive:", naive_metrics)
print("Rolling Moving Avg:", moving_avg_metrics)

##Chosen ML Model (Random Forest)
print("\nTraining Random Forest")

features = [
    'temp_edm_C',
    'rel_hum_edm_pct',
    'wind_edm_kmh',
    'hour',
    'day_of_week',
    'month',
    'is_weekend',
    'year',
    f'{target}_lag_1',
    f'{target}_lag_24',
    f'{target}_lag_168',
    f'{target}_rolling_mean_24',
    f'{target}_rolling_mean_168',
    f'{target}_rolling_std_24',
    f'{target}_rolling_std_168'
]

rf_results, rf_model = train_rand_forest(df, features, target, horizon=1)

# Evaluate Random forest for comparison
rf_mae = (rf_results["Actual"] - rf_results["Prediction"]).abs().mean()
rf_mape = (abs((rf_results["Actual"] - rf_results["Prediction"]) / rf_results["Actual"])).mean() * 100

print(f"\nRandom Forest results: {rf_mae:.2f}, MAPE: {rf_mape:.2f}%")

#Save outputs
output_dir = Path("reports/results")
output_dir.mkdir(parents=True, exist_ok=True)

rf_results.to_csv(output_dir / "rf_results.csv", index=False)

print(f"\nResults saved to {output_dir}")
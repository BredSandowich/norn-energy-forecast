# src/ run_pipeline_norn.py --> Runs all of the src files (Norns) in one single entry point for project Norn

import pandas as pd
from pathlib import Path
import yaml
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

#Import modular src/ files
from data.data_prep_urd import build_dataset
from features.features_verdandi import prepare_features
from models.models_skuld import (
    seasonal_naive, flat_naive, rolling_moving_avg,  flat_mov_avg,
    holt_winters, simple_ensemble, weighted_ensemble,
    create_lin_reg, create_rand_forest, create_xgboost,
    fit_model, predict_model)
from evaluation.evaluation_orlog import forecast_evaluation


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

baselines = {
    "Seasonal Naive": seasonal_naive(y_train, horizon),
    "Rolling Moving Avg": rolling_moving_avg(y_train, horizon = horizon),
    "Flat Naive": flat_naive(y_train, horizon),
    "Flat Moving Avg": flat_mov_avg(y_train, horizon)
}
        
baseline_metrics = {}
for name, forecast in baselines.items():
    metrics = forecast_evaluation(y_test.values, forecast.values)
    baseline_metrics[name] = metrics
    print(f"\n{name} Metrics:")
    for k,v in metrics.items():
        print(f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}")


##Machine Learning Models
print("\nTraining Machine Learning models")

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
    f'{target}_rolling_std_168',
    'HDD_edm',
    "CDD_edm"
]

x_train = train[features]
y_train_series = train[target]
X_test = test[features]
Y_test_series = test[target]

ml_models = {
    "Linear Regression": create_lin_reg(),
    "Random Forest": create_rand_forest(),
    "XGBoost": create_xgboost(n_estimators = 100, max_depth = 5)
}

ml_results = {}

for name, model in ml_models.items():
    print(f"\nTraining {name}")
    
    #First we fit
    fitted_model = fit_model(model, x_train, y_train_series)
    
    #Then we predict!
    prediction = predict_model(fitted_model, X_test)
    
    #Save them results
    results_df = pd.DataFrame({
        "Datetime": test["Datetime"].values,
        "Actual": Y_test_series.values,
        "Prediction": prediction
    })

    # Evaluate 
    mae = (results_df["Actual"] - results_df["Prediction"]).abs().mean()
    mape = np.mean(np.abs((results_df["Actual"] - results_df["Prediction"]) / np.where(results_df["Actual"] == 0, 1, results_df["Actual"])))* 100

    ml_results[name] = {
        "Results": results_df,
        "Model": fitted_model,
        "MAE": mae,
        "MAPE": mape,
        "Error": Y_test_series.values - prediction,
        "Percent Error": ((Y_test_series.values - prediction)/Y_test_series.values)*100
    }


    print(f"{name} results: MAE={mae:.2f}, MAPE={mape:.2f}%")



#Save outputs
output_dir = Path("reports/results")
output_dir.mkdir(parents=True, exist_ok=True)

#Baselines
for name, forecast in baselines.items():
    pd.DataFrame({
        "Datetime": test["Datetime"].values,
        "Actual": Y_test_series.values,
        "Prediction": forecast.values,
        "Error": Y_test_series.values - forecast.values,
        "Percent Error": ((Y_test_series.values - forecast.values)/Y_test_series.values)*100}).to_csv(output_dir / f"{name.replace(' ', '_').lower()}_results.csv", index= False,)
    
#ML Models
#Break out individual results for CSV analysis
for name, result_dict in ml_results.items():
    result_dict = result_dict["Results"]
    
    filename = f"{name.replace(' ', '_').lower()}_results.csv"
    result_dict.to_csv(output_dir /filename, index=False)
    print(f"Saved {name} results -> {filename}")

#Results summary
summary = []

for name, result_dict in ml_results.items():
    summary.append({
        "Model": name,
        "MAE": result_dict["MAE"],
        "MAPE": result_dict["MAPE"]
    })
    
summary_df = pd.DataFrame(summary)
summary_df.to_csv(output_dir / "model_summary.csv", index=False)

print("\nSaved model summary → model_summary.csv")


print(f"\nAll results saved to {output_dir}")


#Incorporate back-testing on models for valuation
from analysis.backtesting_urd import run_walk_forward_validation
from evaluation.evaluation_orlog import plot_error_distribution, plot_error_over_time, plot_model_type_comparison, plot_forecast_sample, calculate_kpis

baseline_models = {
    "Seasonal Naive": seasonal_naive,
    "Rolling Moving Avg": rolling_moving_avg,
    "Flat Naive": flat_naive,
    "Flat Moving Avg": flat_mov_avg}

print("Running walk-forward backtesting!")
wf_results = run_walk_forward_validation(df=df, features= features, target= target, ml_models = ml_models, baseline_models = baseline_models, fit_model= fit_model, predict_model= predict_model, horizon=24)

kpi_summary = calculate_kpis(wf_results)
wf_results.to_csv(output_dir / "walk_forward_results.csv", index=False)
kpi_summary.to_csv(output_dir / "walk_forward_kpis.csv", index=False)
#print(wf_results)
print(kpi_summary)

best_model = kpi_summary.sort_values("MAE").iloc[0]["Model"]

plot_error_over_time(wf_results)
plot_error_distribution(wf_results)
#plot_forecast_sample(wf_results, model_name= "Random Forest")
plot_forecast_sample(wf_results, model_name=best_model)
plot_model_type_comparison(kpi_summary)
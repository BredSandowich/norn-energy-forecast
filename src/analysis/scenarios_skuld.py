#src/scenarios_skuld.py

import pandas as pd
import matplotlib.pyplot as plt

#Function for running seperate scenarios for analysis
def run_scenario(model, X, scenario_func):
    scenario_X = scenario_func(X.copy())
    predictions = model.predict(scenario_X)
    
    return pd.DataFrame({
        "Datetime": X["Datetime"].values if "Datetime" in X else None,
        "Prediction": predictions
    })


#Scenario analysis for Temperature increase
def temp_increase_scenario(X, delta=5):
    X["temp_edm_C"] += delta
    
    X["HDD_edm"] = (18- X["temp_edm_C"]).clip(lower=0)
    X["CDD_edm"] = (X["temp_edm_C"]-22).clip(lower=0)
    return X

#Scenario for temperature decrease
def temp_decrease_scenario(X, delta=5):
    X["temp_edm_C"] -= delta
    
    X["HDD_edm"] = (18 - X["temp_edm_C"]).clip(lower=0)
    X["CDD_edm"] = (X["temp_edm_C"] - 22).clip(lower=0)
    return X

#Demand shock scenario (ie sudden swings in load)
def demand_spike_scenario(X, target_col, spike_pct = 0.1):
    X["f{target_col}_lag_1"] *= (1 + spike_pct)
    X["f{target_col}_lag_24"] *= (1 + spike_pct)
    return X

#Chaos scenario (combining scenarios)
def extreme_scenario(X, target_col):
    X= temp_increase_scenario(X, delta=15)
    X= demand_spike_scenario(X, target_col, spike_pct=0.15)
    return X


#Plot for scenario analysis visualization
def plot_scenarios(dates, actual, base, scenario_dict):
    plt.figure(figsize= (12,6))
    
    plt.plot(dates, actual, label="Actual", linewidth=2, color="black")
    plt.plot(dates, base, label="Base Forecast", linewidth=2)
    
    for name, preds in scenario_dict.items():
        plt.plot(dates, preds, label=name, linestyle= "--")
        
    plt.legend()
    plt.title("Scenario Analysis: Forecast Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
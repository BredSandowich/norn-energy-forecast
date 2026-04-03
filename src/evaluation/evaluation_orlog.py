#src/evaluation_fate.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Define evaluation metrics
def forecast_evaluation(y_true, y_pred):
    error = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(error / np.where(y_true == 0, 1, y_true))) * 100
    return {
        "Error": error, 
        "Mean Absolute Error": mae, 
        "Root Mean Square Error": rmse, 
        "Mean Absolute Percent Error": mape
        }
    

#Visualizations for backtesting performance
def plot_error_over_time(df):
    df["Error"] = df["Actual"] - df["Prediction"]
    
    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        plt.plot(subset["Datetime"], subset["Error"], label = model)
    
    plt.legend()
    plt.title("Forecast Error Over Time")
    plt.xlabel("Datetime")
    plt.ylabel("Error")
    plt.show()
    
#Model comparison plot
def plot_error_distribution(df):
    df["Absolute Error"] = np.abs(
        df["Actual"] - df["Prediction"]
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Model", y="Absolute Error")
    plt.xticks(rotation=45)
    plt.title("Error Distribution by Model")
    plt.show()
    
def plot_model_type_comparison(kpi_summary):

    plt.figure(figsize=(8, 5))
    sns.barplot(data=kpi_summary, x="Type", y="MAE")
    plt.title("ML vs Baseline Performance")
    plt.show()
    
def plot_forecast_sample(df, model_name, n_points = 200):
    subset = df[df["Model"] == model_name].copy()

    df = df.tail(n_points)
    plt.plot(subset["Datetime"], subset["Actual"], label="Actual")
    plt.plot(subset["Datetime"], subset["Prediction"], label="Prediction")

    plt.title(f"{model_name} Forecast vs Actual")
    plt.xlabel("Datetime")
    plt.ylabel("Load")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    

#KPI calculation
def calculate_kpis(walk_f_results):
    kpi_summary = walk_f_results.groupby(["Model", "Type"]).apply(
        lambda df: pd.Series({
            "Error": np.mean(df["Prediction"] - df["Actual"]),
            "MAE": np.mean(np.abs(df["Actual"] - df["Prediction"])),
            "MAPE": np.mean(
                np.abs((df["Actual"] - df["Prediction"]) / df["Actual"])
            ) * 100,
            "Directional_Accuracy": np.mean(
                np.sign(df["Actual"].diff().dropna()) ==
                np.sign(df["Prediction"].diff().dropna())
            )
        })
    ).reset_index()

    return kpi_summary
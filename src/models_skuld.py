# src/models_skuld.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

#Define evaluation metrics
def forecast_evaluation(y_true, y_pred):
    error = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(error / np.where(y_true == 0, 1, y_true))) * 100
    return {"Error": error, "Mean Absolute Error": mae, "Root Mean Square Error": rmse, "Mean Absolute Percent Error": mape}


#Baseline Forecasts: Seasonal Naive (Rolling last point), Naive Flat (last point method), Rolling Moving Average, Moving Average Flat, Holts winter (triple exponential smoothing), Combination "Ensemble", and Weighted Ensemble
def seasonal_naive(y_train, horizon=24): #Rolling 24 hour
    last_day = y_train.iloc[-24:]
    forecast = pd.Series(np.tile(last_day.values, int(np.ceil(horizon/24)))[:horizon]) 
    return forecast

def flat_naive(y_train, horizon=24): #Last point (baseline - awful forecast for longer range, benchmark line more than anything)
    last_value = y_train.iloc[-1]
    forecast = pd.Series([last_value]*horizon)
    return forecast

def rolling_moving_avg(y_train, window_days=7, horizon=24): 
    window_hours = window_days *24
    last_window = y_train.iloc[-window_hours:]
    last_window_reshaped = last_window.values.reshape(window_days, 24)
    forecast_values = last_window_reshaped.mean(axis=0)
    values = np.tile(forecast_values, int(np.ceil(horizon /24)))[:horizon]
    forecast = pd.Series(values)
    return forecast

def flat_mov_avg(y_train, window=7, horizon=24): #benchmark again more than anything as it produces a flat forecast
    window_hours = window *24 #7 day average for a benchmark forecast
    avg_value = y_train.rolling(window).mean().iloc[-1]
    forecast = pd.Series([avg_value]*horizon)
    return forecast

def holt_winters(y_train, horizon=24):
    model = ExponentialSmoothing( y_train, seasonal = "add", seasonal_periods= 24).fit()
    forecast = model.forecast(horizon)
    return pd.Series(forecast.values)

def simply_ensemble(*forecasts): #Combine 2 forecasts to see if it improves either
    combined = sum(forecasts) / len(forecasts)
    return combined

def weighted_ensemble(forecasts, mapes): #Combine 2 forecasts weighting one more than other to see if it improves either
    inv_mape = [1/m for m in mapes]
    weights = [v/sum(inv_mape) for v in inv_mape]
    combined = sum(f*w for f,w in zip(forecasts, weights))
    return combined


#Machine Learning Forecasts
def walk_forward_validation(df, features, target, model, horizon=1): #Update the model with new data continuously
    predictions, actuals, timestamps = [], [], []
    for i in range(len(df)-horizon):
        train = df.iloc[:i+horizon]
        test = df.iloc[i+horizon:i+horizon+1]
        if len(test)==0:
            break
        
        x_train = train[features]
        y_train = train[target]
        X_test = test[features]
        Y_test = test[target]
        
        model.fit(x_train, y_train)
        prediction = model.predict(X_test)[0]
        
        predictions.append(prediction)
        actuals.append(Y_test.values[0])
        timestamps.append(test["Datetime"].values[0])
    return pd.DataFrame({"Datetime": timestamps, "Actual": actuals, "Prediction": predictions})

#Linear Regression
def train_lin_reg(df, features, target, horizon=1):
    lr_model = LinearRegression()
    results = walk_forward_validation(df,features, target, lr_model, horizon)
    return results, lr_model

#Random Forest Regression
def train_rand_forest(df, features, target, horizon=1, n_estimators = 60, max_depth=10, random_state=42, n_jobs= -1): #Future development is refining the parameters, going with sckit recommended for starters with a bit of manual tinkering
    rf_model = RandomForestRegressor(n_estimators= n_estimators, max_depth=max_depth,random_state=random_state)
    results = walk_forward_validation(df, features, target, rf_model,horizon)
    return results, rf_model

#XGBoost --> Trying XGBoost as I have read it is a good time series forecasting tool for ML
def train_xgboost(df, features, target, horizon=1, **kwargs):
    xgb_model = XGBRegressor(**kwargs)
    results = walk_forward_validation(df, features, target, xgb_model, horizon=horizon)
    return results, xgb_model
# src/models_skuld.py

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def create_linear_reg():
    return LinearRegression()

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
    
    max_full_days = len(y_train) // 24
    if max_full_days==0:
        last_val = y_train.iloc[-1]
        return pd.Series([last_val]*horizon)
    
    window_days = min(window_days, max_full_days)
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

def simple_ensemble(*forecasts): #Combine 2 forecasts to see if it improves either
    combined = sum(forecasts) / len(forecasts)
    return combined

def weighted_ensemble(forecasts, mapes): #Combine 2 forecasts weighting one more than other to see if it improves either
    inv_mape = [1/m for m in mapes]
    weights = [v/sum(inv_mape) for v in inv_mape]
    combined = sum(f*w for f,w in zip(forecasts, weights))
    return combined

##Machine Learning Forecasts

#Linear Regression
def create_lin_reg():
    lr_model = LinearRegression()
    return lr_model

#Random Forest Regression
def create_rand_forest(n_estimators = 60, max_depth=10, random_state=42, n_jobs= -1): #Future development is refining the parameters, going with sckit recommended for starters with a bit of manual tinkering
    rf_model = RandomForestRegressor(n_estimators= n_estimators, max_depth=max_depth,random_state=random_state, n_jobs=n_jobs)
    return rf_model

#XGBoost --> Trying XGBoost as I have read it is a good time series forecasting tool for ML
def create_xgboost(**kwargs):
    xgb_model = XGBRegressor(**kwargs)
    return xgb_model


#Fit and use forecast functions
def fit_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

def predict_model(model, x_test):
    return model.predict(x_test)
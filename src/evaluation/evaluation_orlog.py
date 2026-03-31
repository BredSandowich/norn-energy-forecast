#src/evaluation_fate.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
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
df["rolling_mean_24"] = df["load_edm_mw"].shift(1).rolling(window=24).mean()
df["rolling_mean_168"] = df["load_edm_mw"].shift(1).rolling(window=168).mean()
df["rolling_std_24"] = df["load_edm_mw"].shift(1).rolling(window=24).std()
df["rolling_std_168"] = df["load_edm_mw"].shift(1).rolling(window=168).std()

# Drop rows with NaN values introduced by lag and rolling features
df = df.dropna()

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

#Walk forward validation for rolling evaulation on top models to simulate real forecasting scenario
def walk_forward_validation(df, features, target, model, horizon):
    predictions = []
    actuals = []
    timestamps = []
    
    for i in range(len(df)- horizon):
        train = df.iloc[:i + horizon]
        test = df.iloc[i + horizon:i + horizon + 1]
        
        if len(test) == 0:
            break
        x_train = train[features]
        y_train = train[target]
        
        X_test = test[features]
        Y_test = test[target]
        
        model.fit(x_train, y_train)
        pred = model.predict(X_test)[0]
        
        predictions.append(pred)
        actuals.append(Y_test.values[0])
        timestamps.append(test["DATETIME"].values[0])
    
    return pd.DataFrame({
        'DATETIME': timestamps,
        'Actual': actuals,
        'Predicted': predictions
    })

features = ['temp_edm_C', 'rel_hum_edm_pct', 'wind_edm_kmh', 'hour', 'day_of_week', 'month', 'is_weekend', 'year', 'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168', 'rolling_std_24', 'rolling_std_168']
target = 'load_edm_mw'


#Run walk forward validation with Random Forest
wf_rf_model = walk_forward_validation(edmonton_df, features, target, RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), horizon=1)
wf_rf_results.to_csv('notebooks/rf_walk_forward_results.csv', index=False)
print("\nWalk forward Random Forest walk forward validation results saved to notebooks/rf_walk_forward_results.csv")

#Coefficient importance from Random Forest
importance = pd.Series(wf_rf_model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index)
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('notebooks/rf_feature_importance.png')
print("\nRandom Forest feature importance plot saved to notebooks/rf_feature_importance.png")

# Evaluate Random Forest forecast
wf_mae = mean_absolute_error(wf_rf_model['actual'], wf_rf_model['predicted'])
wf_mape = np.mean(np.abs((wf_rf_model['actual'] - wf_rf_model['predicted']) / wf_rf_model['actual'])) * 100
print(f"Random Forest Walk Forward Validation - MAE: {wf_mae:.2f}, MAPE: {wf_mape:.2f}%")

#Walk forward Random Forest plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(test_df['DATETIME'], wf_rf_model['actual'], label='Actual Load', marker='o')
plt.plot(test_df['DATETIME'], wf_rf_model['predicted'], label=f'Walk Forward Random Forest Forecast (MAPE: {wf_mape:.2f}%)', marker='x')
plt.title('Random Forest Forecast vs Actual Load')
plt.xlabel('DateTime')
plt.ylabel('Load (MW)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/wf_rf_forecast.png')
print("\nRandom Forest forecast plot saved to notebooks/wf_rf_forecast.png")


#Create confidence interval for Random forest predictions using the distribution of predictions from individual trees
all_tree_predictions = np.stack([tree.predict(x_test) for tree in rf_model.estimators_], axis=1)

lower_con_intval = np.percentile(all_tree_predictions, 2.5, axis=0)
upper_con_intval = np.percentile(all_tree_predictions, 97.5, axis=0)

#Add confidence interval to the forecast plot
plt.fill_between(test_df['DATETIME'], lower_con_intval, upper_con_intval, color='gray', alpha=0.3, label='95% Confidence Interval')
plt.legend()
plt.savefig('notebooks/rf_forecast_with_confidence.png')
print("\nRandom Forest forecast with confidence interval plot saved to notebooks/rf_forecast_with_confidence.png")


#Train a Simple Model - Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_forecast = lr_model.predict(x_test)
# Evaluate Linear Regression forecast
lr_results = forecast_evaluation(y_test, lr_forecast)
lr_mae, lr_rmse, lr_mape = lr_results[1], lr_results[2], lr_results[3]
print(f"Linear Regression - MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}, MAPE: {lr_mape:.2f}%")

#Feature importance from Linear Regression (coefficients)
coefficients = pd.Series(lr_model.coef_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients.values, y=coefficients.index)
plt.title('Feature Importance from Linear Regression')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('notebooks/lr_feature_importance.png')
print("\nLinear Regression feature importance plot saved to notebooks/lr_feature_importance.png")

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


#High return ML Model XGBoost
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=82)
xgb_model.fit(x_train, y_train)
xgb_forecast = xgb_model.predict(x_test)

#Evaluation of XGBoost forecast
xgb_results = forecast_evaluation(y_test, xgb_forecast)
xgb_mae, xgb_rmse, xgb_mape = xgb_results[1], xgb_results[2], xgb_results[3]
print(f"XGBoost - MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}, MAPE: {xgb_mape:.2f}%")

#Plot actual vs forecast XGBoost
plt.figure(figsize=(12, 6))
plt.plot(test_df['DATETIME'], y_test, label='Actual Load', marker='o')
plt.plot(test_df['DATETIME'], xgb_forecast, label=f'XGBoost Forecast (MAPE: {xgb_mape:.2f}%)', marker='x')
plt.title('XGBoost Forecast vs Actual Load')
plt.xlabel('DateTime')
plt.ylabel('Load (MW)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/xgb_forecast.png')
print("\nXGBoost forecast plot saved to notebooks/xgb_forecast.png")

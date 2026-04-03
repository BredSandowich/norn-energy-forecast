#src/backtesting_urd.py

import pandas as pd
import numpy as np

# Split dataset for walk forward validation
def create_walk_forward_splits(df, horizon, step =24, min_train_size=24*7):
    splits =[]
    start = min_train_size
    
    while start + horizon <= len(df):
        train_idx = slice(0, start)
        test_idx = slice(start, start + horizon)
        splits.append((train_idx, test_idx))
        start += step
        
        return splits
    
#Walk forward backtesting function to apply to model valuation
def run_walk_forward_validation(
    df, features, target, ml_models, baseline_models, fit_model, predict_model, horizon =24, step =24, min_train_size=24*7):
    splits = create_walk_forward_splits(df, horizon, step, min_train_size)
    
    all_results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        x_train = train[features]
        y_train = train[target]
        X_test = test[features]
        Y_test = test[target]
        
        #Baseline model evaluation
        for name, baseline_funct in baseline_models.items():
            preds = baseline_funct(y_train, horizon)
            
            folds_df = pd. DataFrame({
                "Datetime": test["Datetime"].values,
                "Actual": Y_test.values,
                "Prediction": preds.values,
                "Model": name,
                "Fold": i,
                "Type": "Baseline"
            })
            all_results.append(folds_df)
        
        #Machine Learning Model back testing
        for name, model in ml_models.items():
            fitted = fit_model(model, x_train, y_train)
            preds = predict_model(fitted, X_test)
            
            folds_df = pd. DataFrame({
                "Datetime": test["Datetime"].values,
                "Actual": Y_test.values,
                "Prediction": preds,
                "Model": name,
                "Fold": i
            })
            
            all_results.append(folds_df)
    
    walk_f_results = pd.concat(all_results)
    return walk_f_results



'''
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
    
'''
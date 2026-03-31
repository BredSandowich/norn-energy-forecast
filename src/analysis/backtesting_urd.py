

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
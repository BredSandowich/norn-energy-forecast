#src/features_verdandi

import pandas as pd

#Add time features to dataset
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Datetime"].dt.year
    df["month"] = df["Datetime"].dt.month
    df["day"] = df["Datetime"].dt.day
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["hour"] = df["Datetime"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["day_of_year"] = df["Datetime"].dt.dayofyear
    return df
    
##Add lag features to dataset. 
#Lag 1 chosen because most recent datapoint typically is a strong predictor of immediate future
#Lag 24 chosen to capture seasonality within 24 hours ie) dataset is hourly for load demand
#Lag 168 chosen to capture weekly seasonality ie) 7days x 24hrs
def add_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{target_col}_lag_1"] = df[target_col].shift(1)
    df[f"{target_col}_lag_24"] = df[target_col].shift(24)
    df[f"{target_col}_lag_168"] = df[target_col].shift(168)
    return df

##Create rolling features for analysis
#Rolling mean/std 24 and 168 are for hours/day and hours/week. Can expand if wanted for monthly hours
#Shift is used to not have data leakage as far as model seeing the value it is trying to predict (grrr)
def add_rolling_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{target_col}_rolling_mean_24"] = df[target_col].rolling(24).mean().shift(1)
    df[f"{target_col}_rolling_mean_168"] = df[target_col].rolling(168).mean().shift(1)
    df[f"{target_col}_rolling_std_24"] = df[target_col].rolling(24).std().shift(1)
    df[f"{target_col}_rolling_std_168"] = df[target_col].rolling(168).std().shift(1)
    return df

#Add in some weather features Heating Degree Days (HDD) and Cooling Degree Days (CDD)
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    HDD_base = 18.0
    CDD_base = 22.0
    
    df["HDD_edm"] = (HDD_base - df["temp_edm_C"]).clip(lower=0)
    df["CDD_edm"] = (df["temp_edm_C"] - CDD_base).clip(lower=0)
    
    df["HDD_cgy"] = (HDD_base - df["temp_cgy_C"]).clip(lower=0)
    df["CDD_cgy"] = (df["temp_cgy_C"] - CDD_base).clip(lower=0)
    

#Add all features to copied dataframe for full dataset (for pipeline usage in run_pipeline_verdandi.py)
def prepare_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    
    #Drop rows for lag/rolling that aren't available as to not break the model with gaps in data
    df = df.dropna(subset=[
        f"{target_col}_lag_1",
        f"{target_col}_lag_24",
        f"{target_col}_lag_168",
        f"{target_col}_rolling_mean_24", 
        f"{target_col}_rolling_mean_168",  
        f"{target_col}_rolling_std_24",  
        f"{target_col}_rolling_std_168"
    ])
    return df


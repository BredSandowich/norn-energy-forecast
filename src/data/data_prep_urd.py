# src/data_prep.py
import pandas as pd
from pathlib import Path
import yaml

# Load weather information
def load_weather(raw_dir: Path) -> pd.DataFrame:
    """Load and preprocess weather data from GCWeather CSV."""
    df_weather = pd.read_csv(raw_dir / "historical_weather_datapull.csv", encoding="latin1", low_memory=False)
    
    # Standardize datetime
    df_weather["Datetime"] = pd.to_datetime(df_weather["Date/Time (LST)"], errors="coerce")
    
    # Clean column names
    df_weather = df_weather.rename(columns=lambda x: x.strip())
    df_weather = df_weather.rename(columns={
        "Temp (Â°C)": "temp_C",
        "Rel Hum (%)": "rel_hum_pct",
        "Wind Spd (km/h)": "wind_kmh",
        "City": "city"
    })
    
    # Keep only relevant columns
    df_weather = df_weather[["Datetime", "city", "temp_C", "rel_hum_pct", "wind_kmh"]]
    
    # Aggregate duplicates
    df_weather = df_weather.groupby(["Datetime", "city"], as_index=False).agg({
        "temp_C": "mean",
        "rel_hum_pct": "mean",
        "wind_kmh": "mean"
    })
    
    return df_weather

#Load AESO Information pulled from website as per data/README.md
def load_aeso(proc_dir: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    """Load cleaned AESO load data and trim to start_date."""
    df_load = pd.read_csv(proc_dir / "aeso_load_clean.csv", parse_dates=["Datetime"])
    df_load["Datetime"] = pd.to_datetime(df_load["Datetime"])
    df_load = df_load[df_load["Datetime"] >= start_date]
    df_load = df_load.sort_values("Datetime").reset_index(drop=True)
    return df_load


#Merge weather and load for a final dataset to run modelling on for project
def merge_weather_load(df_weather: pd.DataFrame, df_load: pd.DataFrame) -> pd.DataFrame:
    """Merge weather and load data by city, interpolate and forward-fill missing weather."""
    df_weather = df_weather.sort_values("Datetime")
    
    # Split by city
    df_edm = df_weather[df_weather["city"] == "Edmonton"].copy()
    df_cal = df_weather[df_weather["city"] == "Calgary"].copy()
    
    # Rename columns to be city-specific
    df_edm = df_edm.rename(columns={
        "temp_C": "temp_edm_C",
        "rel_hum_pct": "rel_hum_edm_pct",
        "wind_kmh": "wind_edm_kmh"
    }).drop(columns=["city"])
    
    df_cal = df_cal.rename(columns={
        "temp_C": "temp_cgy_C",
        "rel_hum_pct": "rel_hum_cgy_pct",
        "wind_kmh": "wind_cgy_kmh"
    }).drop(columns=["city"])
    
    # Merge using backward fill to match most recent weather observation
    df_merged = pd.merge_asof(df_load.sort_values("Datetime"), df_edm.sort_values("Datetime"),on="Datetime", direction="backward")
    
    df_merged = pd.merge_asof(df_merged, df_cal[["Datetime", "temp_cgy_C", "rel_hum_cgy_pct", "wind_cgy_kmh"]].sort_values("Datetime"), on="Datetime", direction="backward")
    
    '''# Add time features moved to feature engineering, can uncomment if wanted for data exploration
    df_merged["hour"] = df_merged["Datetime"].dt.hour
    df_merged["day_of_week"] = df_merged["Datetime"].dt.dayofweek
    df_merged["month"] = df_merged["Datetime"].dt.month
    df_merged["is_weekend"] = df_merged["day_of_week"].isin([5,6]).astype(int)
    '''
    
    # Flags for missing weather
    df_merged["edm_weather_missing"] = df_merged[["temp_edm_C", "rel_hum_edm_pct", "wind_edm_kmh"]].isna().any(axis=1).astype(int)
    df_merged["cgy_weather_missing"] = df_merged[["temp_cgy_C", "rel_hum_cgy_pct", "wind_cgy_kmh"]].isna().any(axis=1).astype(int)
    
    # Interpolate gaps in weather
    weather_cols = ["temp_edm_C", "rel_hum_edm_pct", "wind_edm_kmh",
                    "temp_cgy_C", "rel_hum_cgy_pct", "wind_cgy_kmh"]
    
    df_merged = df_merged.sort_values("Datetime")
    df_merged = df_merged.set_index("Datetime")
    
    for col in weather_cols:
        df_merged[col] = df_merged[col].interpolate(method="time", limit=3)
        df_merged[col] = df_merged[col].ffill(limit=3)
    
    df_merged = df_merged.reset_index()
    
    return df_merged

def save_processed(df: pd.DataFrame, proc_dir: Path, filename: str = "modelling_dataset.csv"):
    """Save processed DataFrame to CSV."""
    proc_dir.mkdir(parents=True, exist_ok=True)
    output_path = proc_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")
    print(f"Shape: {df.shape}")


#Production Validation checks 
def validate_dataset(df: pd.DataFrame):
    if not df["Datetime"].is_monotonic_increasing:
        raise ValueError("Datetime is not sorted")
    if not df["Datetime"].is_unique:
        raise ValueError("Duplicate timestamps found")
    
    missing_percentage = df.isna().mean()
    print(f"\nMissing values (%): {missing_percentage}")
    print("Datetime validation")
    print(df["Datetime"].diff().value_counts())
    print("Correlation")
    print(df[["EDMONTON","temp_edm_C"]].corr())

    return df


# =======================
# Main execution
# =======================
if __name__ == "__main__":
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    raw_dir = Path(config["paths"]["raw_data"])
    proc_dir = Path(config["paths"]["processed_data"])
    
    # Load and process data
    df_weather = load_weather(raw_dir)
    df_load = load_aeso(proc_dir, start_date=df_weather["Datetime"].min())
    df_merged = merge_weather_load(df_weather, df_load)
    validate_dataset(df_merged)
    save_processed(df_merged, proc_dir)
    
    
#Wrapper for run_pipeline_norn.py file
def build_dataset(raw_dir: Path, proc_dir: Path) -> pd.DataFrame:
    """Full data pipeline: load -> merge -> validate -> return"""
    
    df_weather = load_weather(raw_dir)
    df_load = load_aeso(proc_dir, start_date=df_weather["Datetime"].min())
    
    df_merged = merge_weather_load(df_weather, df_load)
    df_merged = validate_dataset(df_merged)
    
    df_merged = df_merged.rename(columns={
    "EDMONTON": "load_edm_mw",
    "CALGARY": "load_cgy_mw"
    })
        
    return df_merged
    
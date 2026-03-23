import pandas as pd
from pathlib import Path
import numpy as np
import yaml

## If loading without yaml
#raw_dir = Path("data/raw")

#Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)
raw_dir = Path(config["paths"]["raw_data"])
processed_dir = Path(config["paths"]["processed_data"])
processed_dir.mkdir(parents=True, exist_ok=True)

files = [
    "Hourly-load-by-area-and-region-2011-to-2017-.xlsx",
    "Hourly-load-by-area-and-region-2017-2020.xlsx",
    "Hourly-load-by-area-and-region-May-2020-to-Oct-2023.xlsx",
    "Hourly-load-by-area-and-region-Nov-2023-to-Dec-2024.xlsx",
]

#inspected XLSX file exports from AESO and they open on different tabs
sheets = [1, 1, 0, 0]

dfs = []

for i, (file, sheet) in enumerate(zip(files, sheets)):

    # first file has extra header row
    if i == 0:
        df = pd.read_excel(raw_dir / file, sheet_name=sheet, skiprows=1)
    else:
        df = pd.read_excel(raw_dir / file, sheet_name=sheet)

    #Have to standardize the column names
    df.columns = df.columns.str.strip().str.upper()
    
    #Initialize HOUR_ENDING_RAW a NaN for all files
    df["HOUR_ENDING_RAW"] = np.nan

    #Inspected and need to adjust/pre-clean date time between the four files to create a DATETIME column
    if "DATE" in df.columns and "HOUR ENDING" in df.columns:
        # keep HOUR ENDING as-is for EDA (string)
        df["HOUR_ENDING_RAW"] = df["HOUR ENDING"].astype(str)
        # convert valid numbers for datetime calculation
        df["HOUR_ENDING_NUM"] = pd.to_numeric(df["HOUR ENDING"], errors="coerce")
        # only create datetime where HOUR_ENDING_NUM is valid
        df["DATETIME"] = pd.to_datetime(df["DATE"], errors="coerce") + pd.to_timedelta(df["HOUR_ENDING_NUM"] - 1, unit="h")
    elif "DT_MST" in df.columns:
        df["DATETIME"] = pd.to_datetime(df["DT_MST"], errors="coerce")

    # standardize Edmonton / Calgary columns, keeping only columns of interest (ie Edmonton, Calgary, date and time)
    # rename common variations
    edmonton_cols = [c for c in df.columns if "EDMONTON" in c]
    calgary_cols = [c for c in df.columns if "CALGARY" in c]

    # keep only the first match if multiple
    edmonton_col = edmonton_cols[0] if edmonton_cols else None
    calgary_col = calgary_cols[0] if calgary_cols else None

    # select relevant columns
    keep_cols = ["DATETIME", "HOUR_ENDING_RAW"]
    if edmonton_col: keep_cols.append(edmonton_col)
    if calgary_col: keep_cols.append(calgary_col)
    
    # Make sure all keep_cols exist; if missing, create as NaN
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[keep_cols]

    # Rename columns for consistency
    rename_dict = {}
    if edmonton_col: rename_dict[edmonton_col] = "EDMONTON"
    if calgary_col: rename_dict[calgary_col] = "CALGARY"
    df = df.rename(columns=rename_dict)

    dfs.append(df)

#DataFrame checks
print(f"Date range: {df_load_clean['DATETIME'].min()} to {df_load_clean['DATETIME'].max()}")
print(f"Missing values:\n{df_load_clean.isna().sum()}")
print(f"Duplicates: {df_load_clean.duplicated().sum()}")

# combine all files
df_load_clean = pd.concat(dfs, ignore_index=True)

#Save cleaned files data
## If not using yaml
#processed_dir = raw_dir.parent / "processed"
#processed_dir.mkdir(parents =True, exist_ok=True)

output_path = processed_dir / "aeso_load_clean.csv"
df_load_clean.to_csv(output_path, index=False)

#print(df_load.head())
print(f"AESO load data cleaned and ssaved to {output_path}")
print(f"Rows: {df_load_clean.shape[0]}, Columns: {df_load_clean.shape[1]}")

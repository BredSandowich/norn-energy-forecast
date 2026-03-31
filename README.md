# norn-energy-forecast
Machine learning project to forecast Alberta electricity demand using AESO data.

## Project Overview
Norn Energy Forecast is a time-series forecasting project that predicts hourly electricity demand for Alberta's power grid using:
- Alberta Electric Systems Operator (AESO) load data
- Government of Canada historical weather data
The goal is to approximate a simplified version of forecasting workflows used by:
- Power market analysts
- Energy trading teams

As with other forecasting projects I chose to name the project after the three Norns of Norse mythology.  I just find it fun to represent forecasts with the Norns who represent the past (Urd), present (Verðandi), and the future (Skuld).  To me this mirrors the structure of the workflow digesting historical data (past), analyze the current system state (present), and predict the future electricity demand (future).  With that being said the final forecast portion will be named after Skuld, the Norn of what is yet to come.

I figured this would be a good portfolio practice project after coming across a job posting and investigating Kaggle datasets from Kaggle. Unfortunately, I couldn't remember my Kaggle password so I looked for datasets more closely aligned with the job posting and came across real electricity systems data published by the Alberta Electric System Operator (AESO) as well as weather data from the Government of Canada historical weather data.  I realize there is also weather data on Environment and Climate Change Canada (ECCC) and Meteostat API that I may investigate as well but this project will evolve as I dig into the datasets that I have.  The goal is to approximate a simplified version of forecasting workflows used by power market analysts and energy trading teams.

## The Norns Framework
Inspired by the three Norns of Norse mythology, who govern fate:
- Urðr (Past) → historical data
- Verðandi (Present) → current state & features
- Skuld (Future) → forecasts
Additional concepts:
- Wyrd → interconnected system (validation & relationships)
- Orlog → final outcome (evaluation & results)

This creates a full modeling lifecycle:
1. Urðr → prepare the past (data_prep_urd.py)
2. Verðandi → understand the present (features_verdandi.py)
3. Skuld → predict the future (models_skuld.py)
4. Wyrd → validate relationships (validation_wyrd.py)
5. Urðr (again) → test history (backtesting_urd.py)
6. Skuld (again) → explore futures (scenarios_skuld.py)
7. Orlog → evaluate outcomes (evaluation_orlog.py)
8. Norn → orchestrate everything (run_pipeline_norn.py)

## Goals

- Download and clean Alberta system load data.
- Explore seasonality, demand patterns, and trends.
- Perform feature engineering for time-series energy data using Python.
- Build baseline and Machine Learning models
- Forecast short-term (hourly) electricity load demand.
- Focus on utilizing Python to train machine learning models for hourly load forecasting.
- Develop a modular, production style pipeline

## Dataset Description
###Target Variables
- load_edm_mw (primary)
- load_cgy_mw (optional)
Representing the hourly electricity demand (MW) at time t.

### Features
**Weather**
- Temperature (Celsius °C), humidity, wind speed (Edmonton and Calgary)

**Time Features**
- Hour, day of week, month, year
- Weekend indicator

**Lag Features**
- t-1, t-24, t-168 (to capture hourly, daily, weekly patterns)

**Rolling Features**
- Rolling mean and standard deviation (24h, 168h for diurnal (daily) cycles and weekly seasonality)
- Shifted to prevent data leakage (ie - to prevent future information from leaking into current training data)

**Data Quality Flags**
- Missing weather indicators per city (to mark hourly data points that were missing in data pull)


##Data Pipeline (Urðr)
**AESO Load Data**
- Combined from 4 Excel CSV files (2011-2024)
- Standardized datetime formats
- Reindexed to continuous hourly timeline
- Missing valued filled via linear interpolation
**Weather Data**
- Pulled from Government of Canada Climate API
- Hourly observations for: Edmonton International Airport and Calgary International Airport stations
Processing includes:
-Datetime parsing
- Column normalization
- Duplicate aggregation

##Merging Strategy
Weather and load data are merged using:
Python `pd.merge_asof(..., direction="backward")`
To ensure:
- Only past or current weather is used
- Prevents future data leakage

##Missing Data Handling
- Interpolation (limit=3)
- Forward fill (limit=3)
- Missing flags retained as features
Mission:
- Balance data completeness and avoid over smoothing

##Data Load Validation
Pipeline checks:
- Continuous hourly timestamps
- No duplicates
- Monotonic datetime
Printed Sanity checks:
- Time gap validation (1-hour intervals)
- Load vs temperature relationship
Insights:
- Low linear correlation (~ -0.08)
- Clear visual nono-linear U-shaped relationship (expected of energy demand)

##Feature Engineering (Verðandi)
- Time decomposistion (hour, weekday, month, etc)
- Lag features for temporal memory
- Rolling statistics with `.shift(1)` to avoid leakage
Rows with insufficient history are dropped to ensure:
- Clean model inputs
- Stable training

##Models (Skuld)
**Baseline Models**
- Flat Naive (last point) / Moving Average
- Seasonal Naive (t-24 last point)
- Rolling Moving Average
- Holt-Winters

**Machine Learning Models**
- Linear Regression
- Random Forest
- XGBoost

**Training Strategy**
Current:
- Train/test split (last 24 hours)
- Planned:
- Walk-forward validation (backtesting module)

##Evaluation (Orlog)
Metrics used:
- Error
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

## Project Structure

- `config/congfig.yaml` : Configuration to navigate repo filepaths in Python scripts.
- `data/processed` : raw AESO load and GOC weather data (not committed to repo. AESO load data currently requires manual download; future work will include automated ingestion from AESO if available), and processed modelling datasets.
- `notebooks/` : exploratory analysis, feature engineering, model training experiments, and forecast analysis notebooks... used as working files/experimentation files that would be streamlined into final production src/ pipeline files
- `reports/` : figures and generated reports.
- `src/` : reusable Python scripts for data loading and prep, feature engineering, and modelling (as well as forecasting pipelines in future developments).

**Final production ready pipeline project structure**
src/
 data/
   data_prep_urd.py
 features/
   features_verdandi.py
 models/
   models_skuld.py
 evaluation/
   validation_wyrd.py
   evaluation_orlog.py
 analysis/
   backtesting_urd.py
   scenarios_skuld.py
 pipeline/
  run_pipeline_urd.py


## Getting started

```bash
1. git clone <repo>
2. `cd into project folder`
3. python -m venv .venv
4. source .venv/bin/activate #For Windows: .venv\Scripts\activate
5. pip install -r requirements.txt

Run the full pipeline:
`python src/pipeline/run_pipeline_norn.py

## Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
requests
```

## Results
|     Model       |   MAE    |   MAPE  |
| --------------- | -------- | -------- |
| Seasonal Naive  |  33.73   |  2.09 %  |
| Moving Average  |  87.24   |  5.28 %  |
| Random Forest   |  32.19   |  2.00 %  |
| XGBoost         |  15.75   |  0.86 %  |



## Future development's
- Adjust and tinker with walk-forward backtesting (production grade validation)
- Scenario analysis (extreeme weather, demand spikes)
- Produce a probabilistic demand forecast
- Simulate automated daily forecast updates pipeline
- Cyclical encoding (sin/cos time features)
- Feature importance analysis
- Multi-step forecasting

##Key Insight
Electricity demand shows a non-linear relationship with temperature:
- High demand in cold weather (higher heating)
- High demand in hot weather (more cooling)
Which reinforces:
- Importance of lag features
- Value of weather as a secondary signal

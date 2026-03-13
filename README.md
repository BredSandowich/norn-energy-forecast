# norn-energy-forecast
Machine learning project to forecast Alberta electricity demand using AESO data.

Norn is a time-series forecasting project that predicts hourly electricity demand for Alberta's power grid using machine learning and weather data.

As with other forecasting projects I chose to name the project after the three Norns of Norse mythology.  I just find it fun to represent forecasts with the Norns who represent the past (Urd), present (Verðandi), and the future (Skuld).  To me this mirrors the structure of the workflow digesting historical data (past), analyze the current system state (present), and predict the future electricity demand (future).  With that being said the final forecast will be named Skuld, after the Norn of what is yet to come.

I figured this would be a good portfolio project after coming across a job posting and investigating Kaggle datasets from Kaggle. Unfortunately, I couldn't remember my Kaggle password so I looked for datasets more closely aligned with the job posting and came across real electricity systems data published by the Alberta Electric System Operator (AESO) as well as weather data from Environment and Climate Change Canada (ECCC).  I realize there is also weather data on Meteostat API that I may investigate as well but this project will evolve as I dig into the datasets that I have.  The goal is to approximate a simplified version of forecasting workflows used by power market analysts and energy trading teams.

## Goals

- Download and clean Alberta system load data.
- Explore seasonality, daily/weekly patterns, and trends.
- Perform feature engineering for time-series energy data using Python.
- Build baseline and Machine Learning models for short-term load forecasting.
- Focus on utilizing Pythong to train machine learning models for hourly load forecasting.

## Future add-on's

- Build a realistic electricity demand forecasting pipeline.
- Produce a probabilistic demand forecast.
- Simulate automated daily forecast updates.



## Project Structure

- `data/` (Urd - past) : raw AESO load and GOC weather data (not committed to repo), and processed modelling datasets.
- `notebooks/` (Verðandi & Skuld – present & future) : exploratory analysis, feature engineering, model training experiments, and forecast analysis notebooks..
- `src/` : reusable Python scripts for data loading and prep, feature engineering, and modelling (as well as forecasting pipelines in future developments).
- `models/` : serialized model artifacts
- `forecasts/` : generated forecast outputs
= `scripts/` : utility scripts
- `reports/` : figures and generated reports.

## Getting started

```bash
1. git clone repo
2. `cd into project folder`
3. python -m venv .venv
4. source .venv/bin/activate #For Windows: .venv\Scripts\activate
5. pip install -r requirements.txt

## Dependencis
```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
requests
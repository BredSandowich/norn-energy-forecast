# Raw AESO Load Data

These files are required for reproducing the project:

| File name | Source URL | Notes |
|-----------|------------|-------|
| Hourly-load-by-area-and-region-2011-to-2017-.xlsx | [AESO Data Requests](https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-load-by-area-and-region/) | Download manually |
| Hourly-load-by-area-and-region-2017-2020.xlsx | same | ... |
| Hourly-load-by-area-and-region-May-2020-to-Oct-2023.xlsx | same | ... |
| Hourly-load-by-area-and-region-Nov-2023-to-Dec-2024.xlsx | same | ... |

**Instructions:**
1. Go to the AESO data request URL.
2. Click each file hyperlink to download it.
3. Save the files in `data/raw/`.
4. Can use note notebooks/data- Urd/ 02-merge-load-weather.py to clean own AESO load files
5. (Optional) Run: python src/data_prep_urd.py to consolidate into one working file (can use and the pre-cleaned AESO files in repo)
6. run_pipeline_norn.py utilizes data_prep_urd.py functions and pre-saved cleaned dataset file saved in repo

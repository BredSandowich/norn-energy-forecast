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
4. Run: python src/data_prep.py to consolidate into one working file.

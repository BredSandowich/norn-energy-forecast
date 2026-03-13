import pandas as pd
from pathlib import Path

raw_director = Path("../data/raw")

files = [
    raw_dir / "Hourly-load-by-area-and-region-2011-to-2017-.xls",
    raw_dir / "Hourly-load-by-area-and-region-2017-2020.xls",
    raw_dir / "Hourly-load-by-area-and-region-May-2020-to-Oct-2023.xls",
    raw_dir / "Hourly-load-by-area-and-region-Nov-2023-to-Dec-2024.xls",
]

dfs = [pd.read_csv(f) for f in files]
load = pd.concat(dfs, ignore_index=True)
import requests
import pandas as pd
from pathlib import Path

#Load AESO data (download CSV load data)
raw_dir = Path("../data/raw")
df_aeso = pd.read_csv()
raw_dir.mkdir(parents=True, exist_ok=True)

url = "https://api.aeso.ca/your-endpoint-here" #See AESO API docs for data pulls info
headers = {"Accept": "application/json"}

r = requests.get(url, headers=headers)
data = r.json()

df = pd.json_normalize(data["some_key"])
df.to_csv(raw_dir / "aeso_load_raw.csv", index=False)
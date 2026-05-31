import pandas as pd
from entsoe import EntsoePandasClient

API_KEY = "322c9ed8-e236-4715-845f-1ee24f8d1fd1"
client = EntsoePandasClient(api_key=API_KEY)

start = pd.Timestamp("2025-02-01", tz="Europe/Berlin")
end   = pd.Timestamp("2025-05-01", tz="Europe/Berlin")

# Germany/Luxembourg bidding zone
ts = client.query_load("DE_LU", start=start, end=end)

df = ts.reset_index()
df.columns = ["timestamp", "load_mw"]
df.to_csv("data/grid_load.csv", index=False)

print(f"Saved {len(df)} rows")
print(df.head())
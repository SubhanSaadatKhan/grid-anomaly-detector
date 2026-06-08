import os
import pandas as pd
from entsoe import EntsoePandasClient

API_KEY = os.environ["ENTSOE_API_KEY"]
client = EntsoePandasClient(api_key=API_KEY)

# Load existing data
existing = pd.read_csv("data/grid_load.csv")
existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

# Fetch from last timestamp to now
last = existing["timestamp"].max()
start = last.tz_convert("Europe/Berlin")
end   = pd.Timestamp.now(tz="Europe/Berlin")

if (end - start).total_seconds() < 3600:
    print("Less than 1 hour since last update, skipping.")
else:
    print(f"Fetching data from {start} to {end}")
    ts = client.query_load("DE_LU", start=start, end=end)
    new_df = ts.reset_index()
    new_df.columns = ["timestamp", "load_mw"]

    # Append and deduplicate
    combined = pd.concat([existing, new_df]).drop_duplicates(
        subset="timestamp", keep="last"
    ).sort_values("timestamp").reset_index(drop=True)

    combined.to_csv("data/grid_load.csv", index=False)
    print(f"Updated: {len(combined)} total rows ({len(combined) - len(existing)} new)")
import pandas as pd
import sqlite3
import time

# Setup SQLite
conn = sqlite3.connect("data/anomalies.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS readings (
        timestamp TEXT,
        load_mw REAL,
        rolling_mean REAL,
        rolling_std REAL,
        z_score REAL,
        is_anomaly INTEGER
    )
""")
conn.commit()

# Load data
df = pd.read_csv("data/grid_load.csv", parse_dates=["timestamp"])

WINDOW = 72
THRESHOLD = 2.5

print("Starting stream simulation...")

for i in range(len(df)):
    window_data = df["load_mw"].iloc[max(0, i - WINDOW):i]

    if len(window_data) < WINDOW:
        continue  # not enough data yet

    mean = window_data.mean()
    std  = window_data.std()
    z    = (df["load_mw"].iloc[i] - mean) / std if std > 0 else 0
    is_anomaly = 1 if abs(z) > THRESHOLD else 0

    cursor.execute("""
        INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?)
    """, (
        str(df["timestamp"].iloc[i]),
        df["load_mw"].iloc[i],
        mean, std, z, is_anomaly
    ))
    conn.commit()

    if is_anomaly:
        print(f"ANOMALY at {df['timestamp'].iloc[i]} | load={df['load_mw'].iloc[i]:.0f} MW | z={z:.2f}")

    time.sleep(0.01)  # simulate real-time, 0.01s per point so it finishes fast

conn.close()
print("Stream complete. Database written to data/anomalies.db")
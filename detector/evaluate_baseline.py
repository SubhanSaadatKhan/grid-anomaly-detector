import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# Load from DB
conn = sqlite3.connect("data/anomalies.db")
df = pd.read_sql("SELECT * FROM readings", conn)
conn.close()

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["day_of_week"] = df["timestamp"].dt.day_name()
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month_name()

total = len(df)
total_anomalies = df["is_anomaly"].sum()

print("=" * 50)
print(f"Total readings:    {total}")
print(f"Total anomalies:   {total_anomalies}")
print(f"Overall flag rate: {100 * total_anomalies / total:.2f}%")
print("=" * 50)

# --- Problem 1: Weekly seasonality bias ---
print("\nPROBLEM 1: Anomalies by day of week")
print("(Expect Saturday/Sunday/Monday to dominate)")
dow = (df[df["is_anomaly"] == 1]
       .groupby("day_of_week")
       .size()
       .reindex(["Monday","Tuesday","Wednesday",
                 "Thursday","Friday","Saturday","Sunday"])
)
print(dow.to_string())

# --- Problem 2: Holiday clustering ---
print("\nPROBLEM 2: Anomalies by month")
print("(Expect April to spike due to Easter)")
month = (df[df["is_anomaly"] == 1]
         .groupby("month")
         .size()
)
print(month.to_string())

# --- Problem 3: False positives on known stable period ---
print("\nPROBLEM 3: False positive rate on known stable week")
print("(Picking a normal working week: March 10-14)")
stable = df[
    (df["timestamp"] >= "2025-03-10") &
    (df["timestamp"] <= "2025-03-14")
]
fp_rate = stable["is_anomaly"].mean()
print(f"False positive rate: {fp_rate:.2%}")
print(f"Flags in stable week: {stable['is_anomaly'].sum()} / {len(stable)}")

# --- Problem 4: Consecutive flags (drift instability) ---
print("\nPROBLEM 4: Consecutive anomaly streaks")
print("(Long streaks = model confused by drift, not real events)")
df["streak_id"] = (df["is_anomaly"] != df["is_anomaly"].shift()).cumsum()
streaks = (df[df["is_anomaly"] == 1]
           .groupby("streak_id")
           .size()
           .sort_values(ascending=False)
)
print(f"Longest streak: {streaks.iloc[0]} consecutive flags")
print(f"Streaks longer than 3: {(streaks > 3).sum()}")

# --- Plot: where anomalies cluster by hour ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

dow_plot = df[df["is_anomaly"] == 1]["day_of_week"].value_counts().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)
axes[0].bar(dow_plot.index, dow_plot.values, color="tomato")
axes[0].set_title("Anomalies by Day of Week")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=30)

hour_plot = df[df["is_anomaly"] == 1]["hour"].value_counts().sort_index()
axes[1].bar(hour_plot.index, hour_plot.values, color="steelblue")
axes[1].set_title("Anomalies by Hour of Day")
axes[1].set_ylabel("Count")
axes[1].set_xlabel("Hour")

plt.tight_layout()
plt.savefig("data/baseline_failure_analysis.png", dpi=150)
plt.show()

print("\nPlot saved to data/baseline_failure_analysis.png")
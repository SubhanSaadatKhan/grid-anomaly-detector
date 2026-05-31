import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/grid_load.csv", parse_dates=["timestamp"])

# Rolling Z-score
WINDOW = 72        # 72 hourly points = 3 days of context
THRESHOLD = 2.5    # flag anything beyond 2.5 std deviations

df["rolling_mean"] = df["load_mw"].rolling(WINDOW).mean()
df["rolling_std"]  = df["load_mw"].rolling(WINDOW).std()
df["z_score"]      = (df["load_mw"] - df["rolling_mean"]) / df["rolling_std"]
df["is_anomaly"]   = df["z_score"].abs() > THRESHOLD

# Print summary
anomaly_count = df["is_anomaly"].sum()
print(f"Total points: {len(df)}")
print(f"Anomalies flagged: {anomaly_count} ({100*anomaly_count/len(df):.1f}%)")

print("\nTop 10 biggest anomalies:")
print(df[df["is_anomaly"]][["timestamp", "load_mw", "z_score"]]
      .reindex(df[df["is_anomaly"]]["z_score"].abs().nlargest(10).index)
      .to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["timestamp"], df["load_mw"], color="steelblue", linewidth=0.8, label="Load (MW)")
anomalies = df[df["is_anomaly"]]
ax.scatter(anomalies["timestamp"], anomalies["load_mw"],
           color="red", s=20, zorder=5, label=f"Anomalies ({anomaly_count})")
ax.set_title("ENTSO-E Grid Load — Anomaly Detection (Rolling Z-Score)")
ax.set_xlabel("Time")
ax.set_ylabel("Load (MW)")
ax.legend()
plt.tight_layout()
plt.savefig("data/anomaly_plot.png", dpi=150)
plt.show()
print("Plot saved to data/anomaly_plot.png")


import pandas as pd
import numpy as np
import torch
import pickle
import sqlite3
import matplotlib.pyplot as plt
from lstm_model import GridLSTM

# ── 1. Load data and scaler ──────────────────────────────
df = pd.read_csv("data/grid_load.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

with open("data/scaler.pkl", "rb") as f:
    load_scaler = pickle.load(f)

# ── 2. Build features (same as training) ─────────────────
df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["hour_norm"] = df["hour"] / 24.0
df["dow_norm"]  = df["day_of_week"] / 6.0
df["load_scaled"] = load_scaler.transform(df[["load_mw"]])

features = df[["load_scaled", "hour_norm", "dow_norm"]].values

# ── 3. Load trained model ────────────────────────────────
device = torch.device("cpu")
model = GridLSTM(input_size=3, hidden_size=64, num_layers=2, dropout=0.2).to(device)
model.load_state_dict(torch.load("data/lstm_best.pt", map_location=device))
model.eval()

# ── 4. Generate LSTM predictions ─────────────────────────
SEQ_LEN = 672
predictions = []
actuals = []

print("Generating LSTM predictions...")
with torch.no_grad():
    for i in range(SEQ_LEN, len(features)):
        seq = torch.FloatTensor(features[i - SEQ_LEN:i]).unsqueeze(0).to(device)
        pred = model(seq).item()
        predictions.append(pred)
        actuals.append(features[i, 0])

predictions = np.array(predictions)
actuals     = np.array(actuals)

# ── 5. Compute LSTM residuals and anomalies ───────────────
residuals     = actuals - predictions
residual_mean = residuals.mean()
residual_std  = residuals.std()
THRESHOLD     = 3.0

residual_z   = (residuals - residual_mean) / residual_std
lstm_anomaly = np.abs(residual_z) > THRESHOLD

# ── 6. Build LSTM dataframe ───────────────────────────────
df_compare = df.iloc[SEQ_LEN:].copy().reset_index(drop=True)
df_compare["lstm_prediction"] = load_scaler.inverse_transform(predictions.reshape(-1, 1))
df_compare["residual"]        = df_compare["load_mw"] - df_compare["lstm_prediction"]
df_compare["residual_z"]      = residual_z
df_compare["lstm_anomaly"]    = lstm_anomaly.astype(int)

# ── 7. Load Z-score results from SQLite (already computed) 
conn     = sqlite3.connect("data/anomalies.db")
df_zscore = pd.read_sql("SELECT * FROM readings", conn)
conn.close()

df_zscore["timestamp"] = pd.to_datetime(df_zscore["timestamp"], utc=True)

# Merge on timestamp so both detectors cover the same rows
df_valid = df_compare.merge(
    df_zscore[["timestamp", "is_anomaly"]].rename(
        columns={"is_anomaly": "zscore_anomaly"}
    ),
    on="timestamp",
    how="inner"
).reset_index(drop=True)

print(f"Rows after merging both detectors: {len(df_valid)}")

# ── 8. Print comparison ───────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON: Z-Score vs LSTM Residual Detector")
print("=" * 60)

z_total = df_valid["zscore_anomaly"].sum()
l_total = df_valid["lstm_anomaly"].sum()
both    = ((df_valid["zscore_anomaly"] == 1) & (df_valid["lstm_anomaly"] == 1)).sum()
z_only  = ((df_valid["zscore_anomaly"] == 1) & (df_valid["lstm_anomaly"] == 0)).sum()
l_only  = ((df_valid["zscore_anomaly"] == 0) & (df_valid["lstm_anomaly"] == 1)).sum()

print(f"Total readings analysed:     {len(df_valid)}")
print(f"\nZ-Score anomalies:           {z_total} ({100*z_total/len(df_valid):.2f}%)")
print(f"LSTM anomalies:              {l_total} ({100*l_total/len(df_valid):.2f}%)")
print(f"\nAgreed (both flagged):       {both}")
print(f"Z-Score only (LSTM cleared): {z_only}  <- false positives removed by LSTM")
print(f"LSTM only (Z-Score missed):  {l_only}  <- new anomalies LSTM found")

# ── 9. Day of week breakdown ──────────────────────────────
print("\nAnomaly count by day of week:")
print(f"{'Day':<12} {'Z-Score':>8} {'LSTM':>8}")
for day_name, day_num in [("Monday",0),("Tuesday",1),("Wednesday",2),
                           ("Thursday",3),("Friday",4),("Saturday",5),("Sunday",6)]:
    z_day = df_valid[(df_valid["day_of_week"] == day_num) & (df_valid["zscore_anomaly"] == 1)].shape[0]
    l_day = df_valid[(df_valid["day_of_week"] == day_num) & (df_valid["lstm_anomaly"]   == 1)].shape[0]
    print(f"{day_name:<12} {z_day:>8} {l_day:>8}")

# ── 10. False positive rate on known stable week ──────────
stable = df_valid[
    (df_valid["timestamp"] >= "2025-03-10") &
    (df_valid["timestamp"] <= "2025-03-14")
]
z_fp = stable["zscore_anomaly"].mean()
l_fp = stable["lstm_anomaly"].mean()
print(f"\nFalse positive rate on stable week (March 10-14):")
print(f"  Z-Score: {z_fp:.2%}")
print(f"  LSTM:    {l_fp:.2%}")

# ── 11. Consecutive streak analysis ──────────────────────
df_valid["lstm_streak"] = (
    df_valid["lstm_anomaly"] != df_valid["lstm_anomaly"].shift()
).cumsum()
lstm_streaks = (df_valid[df_valid["lstm_anomaly"] == 1]
                .groupby("lstm_streak").size()
                .sort_values(ascending=False))

df_valid["z_streak"] = (
    df_valid["zscore_anomaly"] != df_valid["zscore_anomaly"].shift()
).cumsum()
z_streaks = (df_valid[df_valid["zscore_anomaly"] == 1]
             .groupby("z_streak").size()
             .sort_values(ascending=False))

print(f"\nLongest consecutive streak:")
print(f"  Z-Score: {z_streaks.iloc[0] if len(z_streaks) > 0 else 0}")
print(f"  LSTM:    {lstm_streaks.iloc[0] if len(lstm_streaks) > 0 else 0}")

# ── 12. Plot ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(df_valid["timestamp"], df_valid["load_mw"],
             color="steelblue", linewidth=0.5)
z_anom = df_valid[df_valid["zscore_anomaly"] == 1]
axes[0].scatter(z_anom["timestamp"], z_anom["load_mw"],
                color="red", s=10, zorder=5)
axes[0].set_title(f"Z-Score Detector ({z_total} anomalies, {100*z_total/len(df_valid):.2f}%)")
axes[0].set_ylabel("Load (MW)")

axes[1].plot(df_valid["timestamp"], df_valid["load_mw"],
             color="steelblue", linewidth=0.5)
l_anom = df_valid[df_valid["lstm_anomaly"] == 1]
axes[1].scatter(l_anom["timestamp"], l_anom["load_mw"],
                color="red", s=10, zorder=5)
axes[1].set_title(f"LSTM Residual Detector ({l_total} anomalies, {100*l_total/len(df_valid):.2f}%)")
axes[1].set_ylabel("Load (MW)")

axes[2].plot(df_valid["timestamp"], df_valid["residual"],
             color="gray", linewidth=0.5)
axes[2].axhline(y= residual_std * THRESHOLD, color="red", linestyle="--", alpha=0.5)
axes[2].axhline(y=-residual_std * THRESHOLD, color="red", linestyle="--", alpha=0.5)
axes[2].set_title("LSTM Prediction Residuals (red dashed = anomaly threshold)")
axes[2].set_ylabel("Residual (MW)")
axes[2].set_xlabel("Time")

plt.tight_layout()
plt.savefig("data/comparison_plot.png", dpi=150)
plt.show()

# ── 13. Save full results ─────────────────────────────────
df_valid.to_csv("data/comparison_results.csv", index=False)
print("\nComparison plot saved to data/comparison_plot.png")
print("Full results saved to data/comparison_results.csv")
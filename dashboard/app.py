import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Grid Anomaly Detector", layout="wide")
st.title("ENTSO-E Grid Load — Anomaly Detection")
st.caption("Real-time anomaly detection on German electricity grid data")

# Load comparison results
import os

# Work from repo root regardless of where the app runs
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "comparison_results.csv"))
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── Metrics row ───────────────────────────────────────────
st.subheader("Detection Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Readings", f"{len(df):,}")
col2.metric("Z-Score Anomalies",
            f"{int(df['zscore_anomaly'].sum())}",
            f"{100*df['zscore_anomaly'].mean():.2f}%")
col3.metric("LSTM Anomalies",
            f"{int(df['lstm_anomaly'].sum())}",
            f"{100*df['lstm_anomaly'].mean():.2f}%")
col4.metric("False Positive Reduction",
            f"{100*(1 - df['lstm_anomaly'].sum()/max(df['zscore_anomaly'].sum(),1)):.0f}%")

# ── Comparison chart ──────────────────────────────────────
st.subheader("Z-Score vs LSTM Residual Detector")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=(
        f"Z-Score Detector ({int(df['zscore_anomaly'].sum())} anomalies)",
        f"LSTM Residual Detector ({int(df['lstm_anomaly'].sum())} anomalies)",
        "LSTM Prediction Residuals"
    ),
    vertical_spacing=0.08
)

# Z-Score plot
fig.add_trace(go.Scatter(
    x=df["timestamp"], y=df["load_mw"],
    mode="lines", name="Load",
    line=dict(color="steelblue", width=0.8)
), row=1, col=1)

z_anom = df[df["zscore_anomaly"] == 1]
fig.add_trace(go.Scatter(
    x=z_anom["timestamp"], y=z_anom["load_mw"],
    mode="markers", name="Z-Score Anomaly",
    marker=dict(color="red", size=4)
), row=1, col=1)

# LSTM plot
fig.add_trace(go.Scatter(
    x=df["timestamp"], y=df["load_mw"],
    mode="lines", name="Load",
    line=dict(color="steelblue", width=0.8),
    showlegend=False
), row=2, col=1)

l_anom = df[df["lstm_anomaly"] == 1]
fig.add_trace(go.Scatter(
    x=l_anom["timestamp"], y=l_anom["load_mw"],
    mode="markers", name="LSTM Anomaly",
    marker=dict(color="red", size=4)
), row=2, col=1)

# Residuals plot
fig.add_trace(go.Scatter(
    x=df["timestamp"], y=df["residual"],
    mode="lines", name="Residual",
    line=dict(color="gray", width=0.5)
), row=3, col=1)

residual_std = df["residual"].std()
fig.add_hline(y=residual_std * 3, line_dash="dash",
              line_color="red", opacity=0.5, row=3, col=1)
fig.add_hline(y=-residual_std * 3, line_dash="dash",
              line_color="red", opacity=0.5, row=3, col=1)

fig.update_layout(height=800, showlegend=True,
                  legend=dict(orientation="h"))
fig.update_yaxes(title_text="Load (MW)", row=1)
fig.update_yaxes(title_text="Load (MW)", row=2)
fig.update_yaxes(title_text="Residual (MW)", row=3)

st.plotly_chart(fig, use_container_width=True)

# ── Comparison table ──────────────────────────────────────
st.subheader("Comparison Metrics")

# Stable week FP rate
stable = df[
    (df["timestamp"] >= "2025-03-10") &
    (df["timestamp"] <= "2025-03-14")
]

# Longest streaks
df["z_str"] = (df["zscore_anomaly"] != df["zscore_anomaly"].shift()).cumsum()
z_streaks = (df[df["zscore_anomaly"] == 1].groupby("z_str").size()
             .sort_values(ascending=False))

df["l_str"] = (df["lstm_anomaly"] != df["lstm_anomaly"].shift()).cumsum()
l_streaks = (df[df["lstm_anomaly"] == 1].groupby("l_str").size()
             .sort_values(ascending=False))

metrics = pd.DataFrame({
    "Metric": [
        "Total anomalies flagged",
        "Flag rate",
        "FP rate (stable week)",
        "Longest false alarm streak",
        "Day-of-week bias (Mon vs avg)"
    ],
    "Z-Score": [
        int(df["zscore_anomaly"].sum()),
        f"{100*df['zscore_anomaly'].mean():.2f}%",
        f"{100*stable['zscore_anomaly'].mean():.2f}%",
        int(z_streaks.iloc[0]) if len(z_streaks) > 0 else 0,
        f"{df[df['day_of_week']==0]['zscore_anomaly'].sum()} Mon vs {df['zscore_anomaly'].sum()//7} avg"
    ],
    "LSTM": [
        int(df["lstm_anomaly"].sum()),
        f"{100*df['lstm_anomaly'].mean():.2f}%",
        f"{100*stable['lstm_anomaly'].mean():.2f}%",
        int(l_streaks.iloc[0]) if len(l_streaks) > 0 else 0,
        f"{df[df['day_of_week']==0]['lstm_anomaly'].sum()} Mon vs {df['lstm_anomaly'].sum()//7} avg"
    ]
})

st.dataframe(metrics, use_container_width=True, hide_index=True)

# ── Anomaly detail table ──────────────────────────────────
st.subheader("LSTM Detected Anomalies")
st.dataframe(
    df[df["lstm_anomaly"] == 1][["timestamp", "load_mw", "lstm_prediction", "residual", "residual_z"]]
    .sort_values("residual_z", key=abs, ascending=False)
    .reset_index(drop=True)
    .rename(columns={
        "timestamp": "Timestamp",
        "load_mw": "Actual Load (MW)",
        "lstm_prediction": "LSTM Predicted (MW)",
        "residual": "Residual (MW)",
        "residual_z": "Residual Z-Score"
    }),
    use_container_width=True
)
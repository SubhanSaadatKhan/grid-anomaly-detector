import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go

st.set_page_config(page_title="Grid Anomaly Detector", layout="wide")
st.title("ENTSO-E Grid Load — Live Anomaly Detection")

# Load from SQLite
conn = sqlite3.connect("data/anomalies.db")
df = pd.read_sql("SELECT * FROM readings", conn)
conn.close()

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Total Readings", len(df))
col2.metric("Anomalies Detected", int(df["is_anomaly"].sum()))
col3.metric("Anomaly Rate", f"{100 * df['is_anomaly'].mean():.1f}%")

# Main chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["timestamp"], y=df["load_mw"],
    mode="lines", name="Load (MW)",
    line=dict(color="steelblue", width=1)
))

anomalies = df[df["is_anomaly"] == 1]
fig.add_trace(go.Scatter(
    x=anomalies["timestamp"], y=anomalies["load_mw"],
    mode="markers", name="Anomaly",
    marker=dict(color="red", size=6)
))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Load (MW)",
    legend=dict(orientation="h"),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Anomaly table
st.subheader("Flagged Anomalies")
st.dataframe(
    anomalies[["timestamp", "load_mw", "z_score"]]
    .sort_values("z_score", key=abs, ascending=False)
    .reset_index(drop=True),
    use_container_width=True
)
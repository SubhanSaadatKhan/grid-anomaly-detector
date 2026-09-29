"""Generate the explanatory images used in README.md.

Run from the repo root:  python docs/make_readme_figures.py
Reads data/grid_load.csv and data/comparison_results.csv; writes docs/images/*.png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
TZ = "Europe/Berlin"

# Palette (validated for colour-blind separation)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ACTUAL = "#2a78d6"      # blue   - what the grid actually used
PREDICTED = "#4a3aa7"   # violet - what the AI expected (also dashed)
ALARM = "#e34948"       # red    - an alarm was raised

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titlepad": 10,
})


def gw(ax):
    """Show the y axis in gigawatts (easier to read than 50,000 MW)."""
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f} GW"))


def alarms(ax, df, col):
    hit = df[df[col] == 1]
    ax.scatter(hit["t"], hit["load_mw"], s=46, color=ALARM, edgecolor=SURFACE,
               linewidth=1.5, zorder=5)
    return len(hit)


def load_results():
    df = pd.read_csv(ROOT / "data" / "comparison_results.csv")
    df["t"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
    return df


# ── 1. A typical week of German electricity demand ──────────────────────
def typical_week():
    df = pd.read_csv(ROOT / "data" / "grid_load.csv")
    df["t"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
    # most recent complete Monday-Sunday week
    last_monday = (df["t"].max().normalize() - pd.Timedelta(days=df["t"].max().dayofweek))
    start, end = last_monday - pd.Timedelta(days=7), last_monday
    w = df[(df["t"] >= start) & (df["t"] < end)]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(w["t"], w["load_mw"], color=ACTUAL, linewidth=2)
    weekend = w[w["t"].dt.dayofweek >= 5]
    ax.axvspan(weekend["t"].min(), end, color=GRID, alpha=0.5, linewidth=0)
    ax.set_title(f"One normal week of electricity use in Germany "
                 f"({start:%d %b} – {(end - pd.Timedelta(days=1)):%d %b %Y})")
    gw(ax)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=w["t"].dt.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz=w["t"].dt.tz))
    ax.set_xlim(start, end)

    wd = w[w["t"].dt.dayofweek == 1]                       # Tuesday
    peak, low = wd.loc[wd["load_mw"].idxmax()], wd.loc[wd["load_mw"].idxmin()]
    ax.annotate("Daytime peak:\nfactories, offices, shops", (peak["t"], peak["load_mw"]),
                xytext=(0, 18), textcoords="offset points", ha="center", va="bottom",
                color=INK_2, fontsize=10)
    ax.annotate("Night-time dip:\nmost people asleep", (low["t"], low["load_mw"]),
                xytext=(0, -14), textcoords="offset points", ha="center", va="top",
                color=INK_2, fontsize=10)
    ax.text(weekend["t"].min() + (end - weekend["t"].min()) / 2, ax.get_ylim()[1],
            "Weekend: lower demand", ha="center", va="top", color=INK_2, fontsize=10)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - (ymax - ymin) * 0.15, ymax + (ymax - ymin) * 0.12)
    fig.tight_layout()
    fig.savefig(OUT / "typical_week.png", dpi=150)
    plt.close(fig)


# ── 2. Old vs new detector on an ordinary working week ──────────────────
def old_vs_new(df):
    w = df[(df["t"] >= pd.Timestamp("2025-03-10", tz=TZ)) &
           (df["t"] < pd.Timestamp("2025-03-15", tz=TZ))]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)

    a1.plot(w["t"], w["load_mw"], color=ACTUAL, linewidth=2)
    n1 = alarms(a1, w, "zscore_anomaly")
    a1.set_title(f"Old method (simple statistics): {n1} alarms in a week where nothing went wrong")

    a2.plot(w["t"], w["lstm_prediction"], color=PREDICTED, linewidth=2, linestyle=(0, (4, 3)))
    a2.plot(w["t"], w["load_mw"], color=ACTUAL, linewidth=2)
    n2 = alarms(a2, w, "lstm_anomaly")
    a2.set_title(f"New method (AI forecast): {n2} alarm{'s' if n2 != 1 else ''}")
    last = w.iloc[-1]
    a2.annotate("actual", (last["t"], last["load_mw"]), xytext=(6, 6),
                textcoords="offset points", color=ACTUAL, fontsize=10, fontweight="bold")
    a2.annotate("AI expected", (last["t"], last["lstm_prediction"]), xytext=(6, -12),
                textcoords="offset points", color=PREDICTED, fontsize=10, fontweight="bold")

    for ax in (a1, a2):
        gw(ax)
    a2.xaxis.set_major_locator(mdates.DayLocator(tz=w["t"].dt.tz))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=w["t"].dt.tz))
    a2.set_xlim(w["t"].min(), w["t"].max() + pd.Timedelta(hours=10))
    fig.legend(handles=[
        plt.Line2D([], [], color=ACTUAL, lw=2, label="Actual electricity use"),
        plt.Line2D([], [], color=PREDICTED, lw=2, ls=(0, (4, 3)), label="What the AI expected"),
        plt.Line2D([], [], marker="o", ls="", color=ALARM, markersize=7, label="Alarm raised"),
    ], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "old_vs_new.png", dpi=150)
    plt.close(fig)


# ── 3. What a real alarm looks like ─────────────────────────────────────
def real_alarm(df):
    """Zoom in on the single biggest surprise the AI flagged."""
    lstm = df[df["lstm_anomaly"] == 1]
    top = lstm.loc[lstm["residual"].abs().idxmax()]
    w = df[(df["t"] >= top["t"] - pd.Timedelta(hours=30)) &
           (df["t"] <= top["t"] + pd.Timedelta(hours=18))]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(w["t"], w["lstm_prediction"], color=PREDICTED, linewidth=2, linestyle=(0, (4, 3)))
    ax.plot(w["t"], w["load_mw"], color=ACTUAL, linewidth=2)
    alarms(ax, w, "lstm_anomaly")
    old = int(w.loc[w["t"] == top["t"], "zscore_anomaly"].iloc[0])
    gap = abs(top["residual"]) / 1000
    ax.set_title(f"A real alarm: {top['t']:%A %d %B %Y, %H:%M}")
    ax.annotate(f"Demand suddenly fell {gap:.1f} GW below\nwhat the AI expected - "
                f"about the use of\n{gap * 1e9 / 2000 / 1e6:.1f} million electric kettles.\n"
                f"Old method: {'alarm' if old else 'no alarm'}.",
                (top["t"], top["load_mw"]), xytext=(0.02, 0.97), textcoords="axes fraction",
                va="top", color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK_2, shrinkB=6,
                                connectionstyle="arc3,rad=-0.2"))
    gw(ax)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18], tz=w["t"].dt.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M", tz=w["t"].dt.tz))
    ax.set_xlim(w["t"].min(), w["t"].max())
    ax.legend(handles=[
        plt.Line2D([], [], color=ACTUAL, lw=2, label="Actual electricity use"),
        plt.Line2D([], [], color=PREDICTED, lw=2, ls=(0, (4, 3)), label="What the AI expected"),
        plt.Line2D([], [], marker="o", ls="", color=ALARM, markersize=7, label="Alarm raised"),
    ], loc="lower left", frameon=False, ncol=3)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - (ymax - ymin) * 0.15, ymax + (ymax - ymin) * 0.3)
    fig.tight_layout()
    fig.savefig(OUT / "real_alarm.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results = load_results()
    typical_week()
    old_vs_new(results)
    real_alarm(results)
    print(f"Figures written to {OUT}")

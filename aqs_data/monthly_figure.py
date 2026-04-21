"""
Dashboard-ready monthly fire-season PM2.5 figure.
Extracts the Oct–Mar climatology + 2025 comparison plot from analysis_aqs.py.

Usage:
    from monthly_figure import build_monthly_figure
    fig = build_monthly_figure()
    fig.show()  # or pass to a Dash dcc.Graph
"""

import pandas as pd
import plotly.graph_objects as go
from scipy import stats

MONTHLY_CSV = "monthly_baseline_oct_mar_2000_2024.csv"
EVENT_CSV   = "event_window_daily.csv"
RECENT_ONLY = True

MONTH_ORDER  = [10, 11, 12, 1, 2, 3]
MONTH_LABELS = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}
MONTH_SORT   = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]


def build_monthly_figure(
    monthly_csv: str = MONTHLY_CSV,
    event_csv: str   = EVENT_CSV,
    recent_only: bool = RECENT_ONLY,
) -> go.Figure:
    monthly = pd.read_csv(monthly_csv)
    monthly["month_label"] = monthly["month"].map(MONTH_LABELS)

    event = pd.read_csv(event_csv, parse_dates=["date_local"])
    event["month"] = event["date_local"].dt.month

    baseline = monthly[monthly["season_year"].between(2015, 2024)] if recent_only else monthly
    baseline_start = "2014" if recent_only else "2000"

    clim = (
        baseline
        .groupby(["month", "month_label"])["mean"]
        .agg(clim_mean="mean", clim_std="std", n="count")
        .reset_index()
    )
    clim["se"]    = clim["clim_std"] / clim["n"] ** 0.5
    clim["upper"] = clim["clim_mean"] + clim["se"]
    clim["lower"] = clim["clim_mean"] - clim["se"]
    clim = clim.set_index("month").loc[MONTH_ORDER].reset_index()

    monthly_2025 = (
        event[event["month"].isin(MONTH_ORDER)]
        .groupby("month")["arithmetic_mean"]
        .mean()
        .reset_index()
        .rename(columns={"arithmetic_mean": "mean"})
    )
    monthly_2025["month_label"] = monthly_2025["month"].map(MONTH_LABELS)

    def _pvalue(month_num, val_2025):
        hist = baseline.loc[baseline["month"] == month_num, "mean"].dropna()
        if len(hist) < 3 or pd.isna(val_2025):
            return float("nan")
        return stats.ttest_1samp(hist, popmean=val_2025).pvalue

    monthly_2025["pvalue"] = [
        _pvalue(m, v) for m, v in zip(monthly_2025["month"], monthly_2025["mean"])
    ]
    monthly_2025 = monthly_2025.set_index("month").reindex(MONTH_ORDER).reset_index()

    fig = go.Figure()

    # Individual year lines
    for i, (year, grp) in enumerate(baseline.groupby("season_year")):
        grp = grp.set_index("month").reindex(MONTH_ORDER).reset_index()
        fig.add_trace(go.Scatter(
            x=grp["month_label"],
            y=grp["mean"],
            mode="lines",
            line=dict(color="steelblue", width=1),
            opacity=0.25,
            name=f"Years ({baseline_start}–2024)" if i == 0 else str(year),
            legendgroup="years",
            showlegend=(i == 0),
            hovertemplate=f"<b>{year}</b><br>%{{x}}: %{{y:.2f}} µg/m³<extra></extra>",
        ))

    # ±1 SE band
    fig.add_trace(go.Scatter(
        x=clim["month_label"], y=clim["upper"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=clim["month_label"], y=clim["lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(70,130,180,0.2)",
        name="±1 SE", hoverinfo="skip",
    ))

    # Climatology mean
    fig.add_trace(go.Scatter(
        x=clim["month_label"],
        y=clim["clim_mean"],
        mode="lines+markers",
        line=dict(color="steelblue", width=2.5),
        name=f"Mean ({baseline_start}–2024)",
        customdata=clim[["clim_mean", "se"]].values,
        hovertemplate="%{x}<br>Mean: %{customdata[0]:.2f} µg/m³<br>±SE: %{customdata[1]:.2f}<extra></extra>",
    ))

    # 2025 line with p-values in hover
    fig.add_trace(go.Scatter(
        x=monthly_2025["month_label"],
        y=monthly_2025["mean"],
        mode="lines+markers",
        line=dict(color="crimson", width=3),
        marker=dict(size=8),
        name="2025",
        customdata=monthly_2025[["mean", "pvalue"]].values,
        hovertemplate="%{x} 2025<br>Mean: %{customdata[0]:.2f} µg/m³<br>p-value (vs hist.): %{customdata[1]:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="LA County PM2.5 — Oct–Mar Fire Season",
            subtitle=dict(text=f"Thin blue: individual years ({baseline_start}–2024)  |  Band: mean ±1 SE  |  Red: 2025"),
        ),
        xaxis=dict(title="Month", categoryorder="array", categoryarray=MONTH_SORT),
        yaxis=dict(title="PM2.5 (µg/m³)"),
        width=750, height=480,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


if __name__ == "__main__":
    import os, webbrowser
    fig = build_monthly_figure()
    out = os.path.abspath("pm25_fire_season.html")
    fig.write_html(out)
    webbrowser.open(f"file://{out}")
    print(f"Saved → {out}")

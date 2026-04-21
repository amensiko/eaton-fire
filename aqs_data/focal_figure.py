"""
Dashboard-ready focal-station PM2.5 timeseries (Plot 5 from analysis_aqs.py).
Legend entries use station name with direction and resolution in parentheses.

Usage:
    from focal_figure import build_focal_figure
    fig = build_focal_figure()
    fig.show()  # or pass to a Dash dcc.Graph
"""

import pandas as pd
import plotly.graph_objects as go

EVENT_CSV    = "event_window_daily.csv"
SITE_CSV     = "event_window_site_specific.csv"
MONTHLY_CSV  = "monthly_baseline_oct_mar_2000_2024.csv"
RECENT_ONLY  = True

TS_START = "2025-01-01"
TS_END   = "2025-03-31"

MONTH_ORDER  = [10, 11, 12, 1, 2, 3]
MONTH_LABELS = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}

# site_number (int) → display name, for stations missing local_site_name in site-specific CSV
SITE_NUMBER_MAP = {
    16:   "Glendora",
    4010: "North Hollywood (NOHO)",
}

# (source, site_key, direction, resolution, dash_style, color)
FOCAL = [
    ("hourly", "Glendora",                     "E",   "hourly", "solid", "#1f77b4"),
    ("hourly", "North Hollywood (NOHO)",        "W",   "hourly", "solid", "#ff7f0e"),
    ("daily",  "Los Angeles-North Main Street", "SSW", "daily",  "dash",  "#2ca02c"),
    ("daily",  "Pasadena",                      "S",   "3-day",  "dot",   "#d62728"),
]


def build_focal_figure(
    event_csv: str   = EVENT_CSV,
    site_csv: str    = SITE_CSV,
    monthly_csv: str = MONTHLY_CSV,
    recent_only: bool = RECENT_ONLY,
    ts_start: str    = TS_START,
    ts_end: str      = TS_END,
) -> go.Figure:
    # ── Load data ─────────────────────────────────────────────────────────────
    monthly = pd.read_csv(monthly_csv)
    monthly["month_label"] = monthly["month"].map(MONTH_LABELS)

    event = pd.read_csv(event_csv, parse_dates=["date_local"])
    event["month"] = event["date_local"].dt.month

    site_raw = pd.read_csv(site_csv)
    site_raw["datetime"] = pd.to_datetime(
        site_raw["date_local"] + " " + site_raw["time_local"]
    )

    baseline = monthly[monthly["season_year"].between(2015, 2024)] if recent_only else monthly
    baseline_start = "2014" if recent_only else "2000"

    # ── Climatology (monthly mean ± SE, for horizontal reference bands) ───────
    clim = (
        baseline
        .groupby("month")["mean"]
        .agg(clim_mean="mean", clim_std="std", n="count")
        .reset_index()
    )
    clim["se"]    = clim["clim_std"] / clim["n"] ** 0.5
    clim["upper"] = clim["clim_mean"] + clim["se"]
    clim["lower"] = clim["clim_mean"] - clim["se"]

    # ── Hourly and daily source frames ────────────────────────────────────────
    site_raw["site_name"] = site_raw["site_number"].map(SITE_NUMBER_MAP).fillna(
        site_raw["site_number"].astype(str)
    )

    ts = site_raw[
        (site_raw["sample_duration"] == "1 HOUR") &
        site_raw["datetime"].between(pd.Timestamp(ts_start), pd.Timestamp(ts_end + " 23:59"))
    ].copy()

    daily_raw = event[
        (event["sample_duration"] == "24 HOUR") &
        event["date_local"].between(pd.Timestamp(ts_start), pd.Timestamp(ts_end))
    ].copy()
    daily_plot = (
        daily_raw
        .groupby(["local_site_name", "date_local"])["arithmetic_mean"]
        .mean()
        .reset_index()
    )

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Historical mean ± SE as flat monthly segments
    x_mean, y_mean = [], []
    x_upper, y_upper = [], []
    x_lower, y_lower = [], []

    for month_start in pd.date_range(ts_start, ts_end, freq="MS"):
        m = month_start.month
        row = clim[clim["month"] == m]
        if row.empty:
            continue
        cr = row.iloc[0]
        x0 = max(month_start, pd.Timestamp(ts_start))
        x1 = min(month_start + pd.DateOffset(months=1),
                 pd.Timestamp(ts_end) + pd.Timedelta(days=1))
        for x_lst, y_lst, val in [
            (x_mean,  y_mean,  cr["clim_mean"]),
            (x_upper, y_upper, cr["upper"]),
            (x_lower, y_lower, cr["lower"]),
        ]:
            x_lst += [x0, x1, None]
            y_lst += [val, val, None]

    fig.add_trace(go.Scatter(
        x=x_mean, y=y_mean, mode="lines",
        line=dict(color="rgba(80,80,80,0.5)", width=1.2),
        name=f"Hist. mean ±1 SE ({baseline_start}–2024)",
        legendgroup="clim",
        hoverinfo="skip",
    ))
    for x_lst, y_lst in [(x_upper, y_upper), (x_lower, y_lower)]:
        fig.add_trace(go.Scatter(
            x=x_lst, y=y_lst, mode="lines",
            line=dict(color="rgba(80,80,80,0.35)", width=1, dash="dot"),
            showlegend=False, legendgroup="clim", hoverinfo="skip",
        ))

    # Focal station traces
    for source, site_key, direction, resolution, dash, color in FOCAL:
        legend_label = f"{site_key} ({direction}, {resolution})"

        if source == "hourly":
            grp = ts[ts["site_name"] == site_key].sort_values("datetime")
            x_col, y_col = "datetime", "pm25"
            hover_fmt = "%b %d %H:%M"
            mode = "lines"
        else:
            grp = daily_plot[daily_plot["local_site_name"] == site_key].sort_values("date_local")
            x_col, y_col = "date_local", "arithmetic_mean"
            hover_fmt = "%b %d"
            mode = "lines+markers"

        fig.add_trace(go.Scatter(
            x=grp[x_col],
            y=grp[y_col],
            mode=mode,
            name=legend_label,
            connectgaps=False,
            line=dict(width=2, color=color, dash=dash),
            marker=dict(size=5, color=color) if mode == "lines+markers" else {},
            hovertemplate=f"<b>{legend_label}</b><br>%{{x|{hover_fmt}}}<br>%{{y:.1f}} µg/m³<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"LA County PM2.5 — Focal Stations ({ts_start} to {ts_end})",
            subtitle=dict(text="E/W: hourly BAM  |  SSW: BAM daily avg  |  S: FRM 3-day"),
        ),
        xaxis=dict(title="Date", tickformat="%b %d"),
        yaxis=dict(title="PM2.5 (µg/m³)"),
        width=900, height=500,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


if __name__ == "__main__":
    import os, webbrowser
    fig = build_focal_figure()
    out = os.path.abspath("pm25_focal_stations_dash.html")
    fig.write_html(out)
    webbrowser.open(f"file://{out}")
    print(f"Saved → {out}")

# Dashboard script for air quality tab. Creates 3 figures:
#.   1:  map of stations and fire boundary
#.   2:  historical lookback analysis 
#.   3:  timeseries of 4 closest stations Jan-Mar 2025

# Uses data in eaton-fire/dashboard/helpers/data/airquality

import json
import math

import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point, shape, MultiPolygon
from scipy import stats

"""
Dashboard-ready map of AQS monitors relative to the Eaton Fire perimeter.
Reads only local files — no network calls.

Prerequisites:
  - site_names.csv          (monitor locations, already present)
  - eaton_fire_extent.geojson  (run fetch_fire_perimeter.py once to create)

Usage:
    from map_figure import build_map_figure
    fig = build_map_figure()
    fig.show()  # or pass to a Dash dcc.Graph
"""

PERIMETER_PATH = "eaton_fire_extent.geojson"
MONITORS_PATH  = "site_names.csv"


def _load_perimeter(path: str):
    with open(path) as f:
        data = json.load(f)
    if data["type"] == "FeatureCollection":
        geom = max((shape(f["geometry"]) for f in data["features"]), key=lambda g: g.area)
    else:
        geom = shape(data["geometry"])
    return geom


def _geom_to_rings(geom):
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    rings = []
    for poly in polys:
        coords = list(poly.exterior.coords)
        rings.append(([c[1] for c in coords], [c[0] for c in coords]))
    return rings


def _dist_km(fire_geom, lat, lon) -> float:
    pt = Point(lon, lat)
    return 0.0 if fire_geom.contains(pt) else round(pt.distance(fire_geom.boundary) * 111.0, 1)


def _cardinal(cx, cy, lat, lon) -> str:
    dlat = lat - cy
    dlon = (lon - cx) * math.cos(math.radians(cy))
    bearing = math.degrees(math.atan2(dlon, dlat)) % 360
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[round(bearing / 22.5) % 16]


def build_map_figure(
    perimeter_path: str = PERIMETER_PATH,
    monitors_path: str = MONITORS_PATH,
) -> go.Figure:
    fire_geom = _load_perimeter(perimeter_path)
    cx, cy = fire_geom.centroid.x, fire_geom.centroid.y

    monitors = pd.read_csv(monitors_path, dtype={"site_number": str})
    monitors["dist_km"]   = monitors.apply(lambda r: _dist_km(fire_geom, r.latitude, r.longitude), axis=1)
    monitors["direction"] = monitors.apply(lambda r: _cardinal(cx, cy, r.latitude, r.longitude), axis=1)
    monitors = monitors.sort_values("dist_km").reset_index(drop=True)

    fig = go.Figure()

    for i, (lats, lons) in enumerate(_geom_to_rings(fire_geom)):
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(color="orangered", width=2.5),
            fill="toself",
            fillcolor="rgba(255,100,0,0.15)",
            name="Eaton Fire perimeter" if i == 0 else None,
            showlegend=(i == 0),
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scattermapbox(
        lat=monitors["latitude"].tolist(),
        lon=monitors["longitude"].tolist(),
        mode="markers+text",
        marker=dict(size=11, color="#2171b5"),
        text=monitors["local_site_name"].tolist(),
        textposition="top right",
        textfont=dict(size=11, color="#2171b5"),
        name="AQS monitor",
        customdata=monitors[["site_number", "dist_km", "direction"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Site: %{customdata[0]}<br>"
            "Distance: %{customdata[1]:.1f} km %{customdata[2]} of fire"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text="LA County AQS Monitors — Eaton Fire (Jan 2025)",
            subtitle=dict(text="Orange polygon = fire perimeter  |  Hover markers for site details"),
        ),
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=cy, lon=cx),
            zoom=9,
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=60, b=0),
        width=900,
        height=650,
    )

    return fig


if __name__ == "__main__":
    import webbrowser, os
    fig = build_map_figure()
    out = os.path.abspath("map_aqs_eaton_fire.html")
    fig.write_html(out)
    webbrowser.open(f"file://{out}")
    print(f"Map saved → {out}")


"""
Dashboard-ready monthly fire-season PM2.5 figure.
Extracts the Oct–Mar climatology + 2025 comparison plot from analysis_aqs.py.

Usage:
    from monthly_figure import build_monthly_figure
    fig = build_monthly_figure()
    fig.show()  # or pass to a Dash dcc.Graph
"""


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

"""
Dashboard-ready focal-station PM2.5 timeseries (Plot 5 from analysis_aqs.py).
Legend entries use station name with direction and resolution in parentheses.

Usage:
    from focal_figure import build_focal_figure
    fig = build_focal_figure()
    fig.show()  # or pass to a Dash dcc.Graph
"""

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



# END

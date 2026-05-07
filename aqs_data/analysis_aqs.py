"""
Eaton Fire Air Quality Analysis — Exploratory Plots
Reads CSVs produced by pull_aqs_data.py (run from the same directory).
"""

import os
import webbrowser
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from site_config import load_site_name_map, build_color_map

# ── Config ────────────────────────────────────────────────────────────────────
MONTHLY_CSV  = "monthly_baseline_oct_mar_2000_2024.csv"
EVENT_CSV    = "event_window_daily.csv"
SITE_CSV     = "event_window_site_specific.csv"
RECENT_ONLY  = True   # True → 2014–2024 baseline; False → 2000–2024

MONTH_ORDER  = [10, 11, 12, 1, 2, 3]
MONTH_LABELS = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}
MONTH_SORT   = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]

# ── Timeseries plot window (Plot 3) ───────────────────────────────────────────
TS_START = "2025-01-01"   # inclusive
TS_END   = "2025-03-31"   # inclusive

# ── Load ──────────────────────────────────────────────────────────────────────
monthly = pd.read_csv(MONTHLY_CSV)
monthly["month_label"] = monthly["month"].map(MONTH_LABELS)

event = pd.read_csv(EVENT_CSV, parse_dates=["date_local"])
event["month"] = event["date_local"].dt.month

# ── Site name map & global color palette ──────────────────────────────────────
SITE_NAME_MAP  = load_site_name_map()
SITE_COLOR_MAP = build_color_map()

# ── 2025 monthly means (Oct 2024 – Mar 2025) ─────────────────────────────────
monthly_2025 = (
    event[event["month"].isin(MONTH_ORDER)]
    .groupby("month")["arithmetic_mean"]
    .mean()
    .reset_index()
    .rename(columns={"arithmetic_mean": "mean"})
)
monthly_2025["month_label"] = monthly_2025["month"].map(MONTH_LABELS)

# ── Climatology: mean ± SE across season years ───────────────────────────────
# season_year 2015 covers Oct 2014 – Mar 2015, so RECENT_ONLY spans 2014–2024
baseline = monthly[monthly["season_year"].between(2015, 2024)] if RECENT_ONLY else monthly
clim = (
    baseline
    .groupby(["month", "month_label"])["mean"]
    .agg(clim_mean="mean", clim_std="std", n="count")
    .reset_index()
)
clim["se"]    = clim["clim_std"] / clim["n"] ** 0.5
clim["upper"] = clim["clim_mean"] + clim["se"]
clim["lower"] = clim["clim_mean"] - clim["se"]

# ── T-test: 2025 monthly mean vs historical distribution ─────────────────────
# One-sample t-test: sample = historical yearly means for each month,
# popmean = 2025 monthly mean. Tests whether 2025 is an outlier of the
# historical distribution (two-sided).
def ttest_2025_vs_hist(month_num, val_2025):
    hist_vals = baseline.loc[baseline["month"] == month_num, "mean"].dropna()
    if len(hist_vals) < 3 or pd.isna(val_2025):
        return float("nan")
    result = stats.ttest_1samp(hist_vals, popmean=val_2025)
    return result.pvalue

monthly_2025["pvalue"] = [
    ttest_2025_vs_hist(m, v)
    for m, v in zip(monthly_2025["month"], monthly_2025["mean"])
]
print(monthly_2025[["month",'pvalue']])

# Sort all dataframes by the display order before plotting
clim         = clim.set_index("month").loc[MONTH_ORDER].reset_index()
monthly_2025 = monthly_2025.set_index("month").loc[MONTH_ORDER].reset_index()

baseline_start = "2014" if RECENT_ONLY else "2000"

# ── Plot 1: Fire season — Oct–Mar ─────────────────────────────────────────────
fig1 = go.Figure()

# Individual year lines (one per season_year, no legend entries)
for i, (year, grp) in enumerate(baseline.groupby("season_year")):
    grp = grp.set_index("month").reindex(MONTH_ORDER).reset_index()
    fig1.add_trace(go.Scatter(
        x=grp["month_label"],
        y=grp["mean"],
        mode="lines",
        line=dict(color="steelblue", width=1),
        opacity=0.25,
        name=str(year),
        legendgroup="years",
        showlegend=(i == 0),        # one legend entry for the whole group
        legendgrouptitle_text=f"Years ({baseline_start}–2024)" if i == 0 else None,
        hovertemplate=f"<b>{year}</b><br>%{{x}}: %{{y:.2f}} µg/m³<extra></extra>",
    ))

# ±1 SE band (upper then lower with fill)
fig1.add_trace(go.Scatter(
    x=clim["month_label"],
    y=clim["upper"],
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
))
fig1.add_trace(go.Scatter(
    x=clim["month_label"],
    y=clim["lower"],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(70,130,180,0.2)",
    name="±1 SE",
    hoverinfo="skip",
))

# Climatology mean line
fig1.add_trace(go.Scatter(
    x=clim["month_label"],
    y=clim["clim_mean"],
    mode="lines+markers",
    line=dict(color="steelblue", width=2.5),
    name=f"Mean ({baseline_start}–2024)",
    customdata=clim[["clim_mean", "se"]].values,
    hovertemplate="%{x}<br>Mean: %{customdata[0]:.2f} µg/m³<br>±SE: %{customdata[1]:.2f}<extra></extra>",
))

# 2025 line
fig1.add_trace(go.Scatter(
    x=monthly_2025["month_label"],
    y=monthly_2025["mean"],
    mode="lines+markers",
    line=dict(color="crimson", width=3),
    marker=dict(size=8),
    name="2025",
    customdata=monthly_2025[["mean", "pvalue"]].values,
    hovertemplate="%{x} 2025<br>Mean: %{customdata[0]:.2f} µg/m³<br>p-value (vs hist.): %{customdata[1]:.3f}<extra></extra>",
))

fig1.update_layout(
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

# ── Plot 2: January by year — mean ──────────────────────────────────────
jan = monthly[monthly["month"] == 1].sort_values("season_year")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=jan["season_year"],
    y=jan["mean"],
    marker_color="steelblue",
    name="Monthly mean",
    hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f} µg/m³<extra></extra>",
))

fig2.update_layout(
    title=dict(
        text="LA County January PM2.5 by Year (2000–2024)",
    ),
    xaxis=dict(title="Year", dtick=1),
    yaxis=dict(title="PM2.5 (µg/m³)"),
    width=750, height=420,
    showlegend=False,
    template="plotly_white",
)

# ── Plot 3: Station timeseries — true hourly BAM ─────────────────────────────
site_raw = pd.read_csv(SITE_CSV)
site_raw["datetime"] = pd.to_datetime(
    site_raw["date_local"] + " " + site_raw["time_local"]
)
site_raw["site_name"] = site_raw["site_number"].map(SITE_NAME_MAP).fillna(
    site_raw["site_number"].astype(str)
)

ts = site_raw[
    (site_raw["sample_duration"] == "1 HOUR") &
    site_raw["datetime"].between(pd.Timestamp(TS_START), pd.Timestamp(TS_END + " 23:59"))
].copy()

# ── Plot 3: Hourly BAM (1 HOUR, Non-FRM) ─────────────────────────────────────
fig3 = go.Figure()

for site_name, grp in ts.groupby("site_name"):
    grp = grp.sort_values("datetime")
    color = SITE_COLOR_MAP.get(site_name, "#333333")
    fig3.add_trace(go.Scatter(
        x=grp["datetime"],
        y=grp["pm25"],
        mode="lines",
        name=site_name,
        connectgaps=False,
        line=dict(width=1.5, color=color),
        hovertemplate=f"<b>{site_name}</b><br>%{{x|%b %d %H:%M}}<br>%{{y:.1f}} µg/m³<extra></extra>",
    ))

fig3.update_layout(
    title=dict(
        text=f"LA County PM2.5 — Hourly Station Timeseries ({TS_START} to {TS_END})",
        subtitle=dict(text="BAM continuous monitors (1-hour, Non-FRM)  |  one line per station"),
    ),
    xaxis=dict(title="Date / Time", tickformat="%b %d"),
    yaxis=dict(title="PM2.5 (µg/m³)"),
    width=900, height=500,
    hovermode="x unified",
    template="plotly_white",
)

# ── Plot 4: Daily ≥24-hour data (FRM + BAM daily) ────────────────────────────
# Source: event_window_daily.csv, sample_duration == "24 HOUR"
# Multiple POCs per site → average per station per date
daily_raw = event[
    (event["sample_duration"] == "24 HOUR") &
    event["date_local"].between(pd.Timestamp(TS_START), pd.Timestamp(TS_END))
].copy()

daily_plot = (
    daily_raw
    .groupby(["local_site_name", "date_local"])["arithmetic_mean"]
    .mean()
    .reset_index()
)

fig4 = go.Figure()

for site_name, grp in daily_plot.groupby("local_site_name"):
    grp = grp.sort_values("date_local")
    color = SITE_COLOR_MAP.get(site_name, "#333333")
    fig4.add_trace(go.Scatter(
        x=grp["date_local"],
        y=grp["arithmetic_mean"],
        mode="lines+markers",
        name=site_name,
        connectgaps=False,
        line=dict(width=2, color=color, dash="dash"),
        marker=dict(size=6, color=color, symbol="circle-open"),
        hovertemplate=f"<b>{site_name}</b><br>%{{x|%b %d}}<br>%{{y:.1f}} µg/m³<extra></extra>",
    ))

fig4.update_layout(
    title=dict(
        text=f"LA County PM2.5 — Daily Station Timeseries ({TS_START} to {TS_END})",
        subtitle=dict(text="24-hour integrated samples (FRM + BAM daily)  |  one line per station"),
    ),
    xaxis=dict(title="Date", tickformat="%b %d"),
    yaxis=dict(title="PM2.5 (µg/m³)"),
    width=900, height=500,
    hovermode="x unified",
    template="plotly_white",
)

# ── Plot 5: Four focal stations ───────────────────────────────────────────────
# Each entry: (source, site_name_key, legend_label, dash_style)
# source "hourly" → ts (site_raw 1-HOUR); source "daily" → daily_plot (event 24-HOUR)
FOCAL = [
    ("hourly", "Glendora",                      "E: hourly",   "solid"),
    ("hourly", "North Hollywood (NOHO)",         "W: hourly",   "solid"),
    ("daily",  "Los Angeles-North Main Street",  "SSW: daily",  "dash"),
    ("daily",  "Pasadena",                       "S: 3-day",    "dot"),
]

fig5 = go.Figure()

# ── Historical mean ± SE lines (one segment per calendar month) ───────────────
# Build x/y arrays with None separators between months so each segment
# stays within its own month boundary.
x_mean, y_mean = [], []
x_upper, y_upper = [], []
x_lower, y_lower = [], []

for month_start in pd.date_range(TS_START, TS_END, freq="MS"):
    m = month_start.month
    clim_row = clim[clim["month"] == m]
    if clim_row.empty:
        continue
    cr = clim_row.iloc[0]
    x0 = max(month_start, pd.Timestamp(TS_START))
    x1 = min(month_start + pd.DateOffset(months=1),
             pd.Timestamp(TS_END) + pd.Timedelta(days=1))
    for x_lst, y_lst, val in [
        (x_mean,  y_mean,  cr["clim_mean"]),
        (x_upper, y_upper, cr["upper"]),
        (x_lower, y_lower, cr["lower"]),
    ]:
        x_lst += [x0, x1, None]
        y_lst += [val, val, None]

clim_line_style = dict(color="rgba(80,80,80,0.5)", width=1.2)

fig5.add_trace(go.Scatter(
    x=x_mean, y=y_mean, mode="lines",
    line=clim_line_style,
    name=f"Hist. mean ±1 SE ({baseline_start}–2024)",
    legendgroup="clim",
    hoverinfo="skip",
))
for x_lst, y_lst in [(x_upper, y_upper), (x_lower, y_lower)]:
    fig5.add_trace(go.Scatter(
        x=x_lst, y=y_lst, mode="lines",
        line=dict(color="rgba(80,80,80,0.35)", width=1, dash="dot"),
        showlegend=False,
        legendgroup="clim",
        hoverinfo="skip",
    ))

focal_frames = []
for source, site_key, label, dash in FOCAL:
    color = SITE_COLOR_MAP.get(site_key, "#333333")
    if source == "hourly":
        grp = ts[ts["site_name"] == site_key].sort_values("datetime")
        x_col, y_col = "datetime", "pm25"
        hover_x_fmt  = "%b %d %H:%M"
    else:
        grp = daily_plot[daily_plot["local_site_name"] == site_key].sort_values("date_local")
        x_col, y_col = "date_local", "arithmetic_mean"
        hover_x_fmt  = "%b %d"

    focal_frames.append(pd.DataFrame({
        "datetime":    grp[x_col].values,
        "pm25":        grp[y_col].values,
        "station":     site_key,
        "label":       label,
        "resolution":  source,
    }))

    mode = "lines+markers" if dash in ("dash", "dot") else "lines"
    fig5.add_trace(go.Scatter(
        x=grp[x_col],
        y=grp[y_col],
        mode=mode,
        name=label,
        connectgaps=False,
        line=dict(width=2, color=color, dash=dash),
        marker=dict(size=5, color=color) if mode == "lines+markers" else {},
        hovertemplate=f"<b>{label}</b><br>%{{x|{hover_x_fmt}}}<br>%{{y:.1f}} µg/m³<extra></extra>",
    ))

focal_csv = pd.concat(focal_frames, ignore_index=True)
focal_csv.to_csv("pm25_focal_stations.csv", index=False)
print(f"Saved pm25_focal_stations.csv ({len(focal_csv):,} rows)")

# ── Daily-resolution export ───────────────────────────────────────────────────
# Hourly stations → mean of all hourly readings per calendar day
# Pasadena (3-day FRM) → linear interpolation to fill gaps between sample days
# LA-NMS → already daily, passed through as-is
daily_pieces = []

for source, site_key, label, _ in FOCAL:
    if source == "hourly":
        raw = ts[ts["site_name"] == site_key].copy()
        raw["date"] = raw["datetime"].dt.normalize()
        piece = (
            raw.groupby("date")["pm25"]
            .mean()
            .rename("pm25")
            .reset_index()
        )
        piece["interpolated"] = False
    else:
        raw = daily_plot[daily_plot["local_site_name"] == site_key].copy()
        raw = raw.set_index("date_local")["arithmetic_mean"].rename("pm25")
        # Reindex to every calendar day in the window, then interpolate
        full_idx = pd.date_range(TS_START, TS_END, freq="D")
        raw = raw.reindex(full_idx)
        interpolated_mask = raw.isna()
        raw = raw.interpolate(method="time")
        piece = raw.reset_index().rename(columns={"index": "date"})
        piece["interpolated"] = interpolated_mask.values

    piece["station"] = site_key
    piece["label"]   = label
    daily_pieces.append(piece)

focal_daily = (
    pd.concat(daily_pieces, ignore_index=True)
    [["date", "station", "label", "pm25", "interpolated"]]
    .sort_values(["station", "date"])
    .reset_index(drop=True)
)
focal_daily.to_csv("pm25_focal_stations_daily.csv", index=False)
print(f"Saved pm25_focal_stations_daily.csv ({len(focal_daily):,} rows)")

fig5.update_layout(
    title=dict(
        text=f"LA County PM2.5 — Focal Stations ({TS_START} to {TS_END})",
        subtitle=dict(text="E/W: hourly BAM  |  SSW: BAM daily avg  |  S: FRM 3-day"),
    ),
    xaxis=dict(title="Date", tickformat="%b %d"),
    yaxis=dict(title="PM2.5 (µg/m³)"),
    width=900, height=500,
    hovermode="x unified",
    template="plotly_white",
)

# ── Save & open ───────────────────────────────────────────────────────────────
for fname, fig in [
    #("pm25_fire_season.html", fig1),
    #("pm25_january_by_year.html", fig2),
    ("pm25_station_timeseries.html", fig3),
    ("pm25_station_daily.html",      fig4),
    ("pm25_focal_stations.html",     fig5),
]:
    out = os.path.abspath(fname)
    fig.write_html(out)
    webbrowser.open(f"file://{out}")

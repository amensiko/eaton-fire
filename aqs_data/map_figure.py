"""
Dashboard-ready map of AQS monitors relative to the Eaton Fire perimeter.
Reads only local files — no network calls.

Prerequisites:
  - site_names.csv          (monitor locations, already present)
  - eaton_fire_perimeter.geojson  (run fetch_fire_perimeter.py once to create)

Usage:
    from map_figure import build_map_figure
    fig = build_map_figure()
    fig.show()  # or pass to a Dash dcc.Graph
"""

import json
import math

import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point, shape, MultiPolygon

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

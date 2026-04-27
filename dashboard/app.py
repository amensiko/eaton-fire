from dash import Dash, html, dcc, Input, Output, dash_table
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import yaml

from helpers.biodiversity import (
    fig_monthly_taxa_with_users,
    load_biodiversity_data,
    period_counts,
    fig_taxa_counts,
    clean_data,
    fig_observation_map_slider,
    make_clean_bigram_table,
    fig_description_counts_time,
    fig_description_counts_total,
)

from helpers.news import (
    load_and_clean_news_data,
    daily_article_counts,
    top_outlets_counts,
    monthly_text_extraction_success,
    article_size_hist,
    monthly_news_coverage_with_quality,
    topic_modelling,
    topic_modelling_llm_table,
    themes_lines,
    themes_heatmap,
    themes_keywords
)

from helpers.airquality import (
    build_map_figure,
    build_monthly_figure,
    build_focal_figure,
)

from helpers.CCF import cross_correlation
from helpers.CCF_utils import load_variable_lookup


BRITE = "https://bootswatch.com/5/brite/bootstrap.min.css"

app = Dash(__name__, external_stylesheets=[BRITE])

# Biodiversity 
bio_data = load_biodiversity_data()
df = clean_data(bio_data['df'])
df_effort, period_summary = period_counts(df)
period_summary["Observations Per User"] = period_summary["Observations Per User"].round(2)
period_summary["Share of Total Observations"] = period_summary["Share of Total Observations"].round(1)
period_summary["Share of Total Users"] = period_summary["Share of Total Users"].round(1)
df_text = df[df["has_description"]].copy()
bigram_table = make_clean_bigram_table(df_text, top_n=20)
bigram_table.columns = ["Top bigram", "Count"]

# News
fixed_df, news_df, report_df = load_and_clean_news_data()
top_outlets_table = (
    report_df["media_name"]
    .fillna("Unknown")
    .value_counts()
    .head(15)
    .reset_index()
)

top_outlets_table.columns = ["Outlet", "Number of articles"]
topic_llm_table = topic_modelling_llm_table()
theme_long = pd.read_csv("helpers/data/news/themes.csv")
theme_terms_table = themes_keywords()

# CCF
ccf_var_lookup = load_variable_lookup()
ccf_variable_options = []
for key, label in ccf_var_lookup.items():
    ccf_variable_options.append(
        {
            "label": label,
            "value": key,
        }
    )
ccf_station_options = [
    {"label": "Average across stations", "value": "average"},
    {"label": "Glendora", "value": "Glendora"},
    {"label": "Los Angeles-North Main Street", "value": "Los Angeles-North Main Street"},
    {"label": "North Hollywood (NOHO)", "value": "North Hollywood (NOHO)"},
    {"label": "Pasadena", "value": "Pasadena"},
]

app.layout = dbc.Container(
    [
        html.Div(
            html.Div(
                html.H1(
                    "Environmental Impacts of the Eaton Fire",
                    className="dashboard-title m-0"
                ),
                className="title-box"
            ),
            className="title-wrapper mb-4"
        ),

        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                label="Welcome",
                                tab_id="welcome",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("Welcome", className="mb-3"),

                                            html.P(
                                                "This dashboard explores environmental impacts of the Eaton Fire through biodiversity observations, air quality measurements, weather data, news coverage, and cross-dataset comparisons."
                                            ),

                                            html.P(
                                                "Use the tabs above to explore how ecological observations, atmospheric conditions, monitoring data, and public reporting changed before, during, and after the fire."
                                            ),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Card(
                                                            dbc.CardBody([
                                                                html.H5("Air Quality"),
                                                                html.P("Explore PM2.5 monitoring stations, historical comparisons, and focal station time series.")
                                                            ]),
                                                            className="welcome-info-card",
                                                        ),
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Card(
                                                            dbc.CardBody([
                                                                html.H5("Biodiversity"),
                                                                html.P("View iNaturalist observations, taxonomic patterns, user effort, maps, and description keywords.")
                                                            ]),
                                                            className="welcome-info-card",
                                                        ),
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Card(
                                                            dbc.CardBody([
                                                                html.H5("Weather"),
                                                                html.P("Examine weather conditions relevant to fire behavior, smoke transport, and recovery context.")
                                                            ]),
                                                            className="welcome-info-card",
                                                        ),
                                                        width=4,
                                                    ),
                                                ],
                                                className="g-4 mt-3",
                                            ),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Card(
                                                            dbc.CardBody([
                                                                html.H5("News Reports"),
                                                                html.P("Analyze media coverage, environmental themes, topic models, and LLM-generated summaries.")
                                                            ]),
                                                            className="welcome-info-card",
                                                        ),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Card(
                                                            dbc.CardBody([
                                                                html.H5("Cross-Correlation"),
                                                                html.P("Compare patterns across datasets to explore possible relationships between weather, air quality, biodiversity, and news attention.")
                                                            ]),
                                                            className="welcome-info-card",
                                                        ),
                                                        width=6,
                                                    ),
                                                ],
                                                className="g-4 mt-3",
                                            ),

                                            html.Hr(className="my-4"),

                                            html.Div(
                                                [
                                                    html.Img(
                                                        src="/assets/uvm_logo.png",
                                                        style={"height": "36px", "marginRight": "12px"}
                                                    ),
                                                    html.Span(
                                                        "Created by Anastasija Mensikova, Will Behm, and Erik Lagerquist"
                                                    ),
                                                ],
                                                className="text-center text-muted mb-0",
                                                style={
                                                    "display": "flex",
                                                    "justifyContent": "center",
                                                    "alignItems": "center",
                                                    "gap": "10px",
                                                },
                                            )
                                        ],
                                        className="p-4"
                                    )
                                ],
                            ),
                            dbc.Tab(
                                label="Air Quality",
                                tab_id="airquality",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("Air Quality", className="mb-3"),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Dropdown(
                                                            id="airquality-plot-dropdown",
                                                            options=[
                                                                {"label": "Monitor Map", "value": "monitor_map"},
                                                                {"label": "Historical Comparison", "value": "historical_comparison"},
                                                                {"label": "Focal Stations", "value": "focal_stations"},
                                                            ],
                                                            value="monitor_map",
                                                            clearable=False,
                                                        ),
                                                        width=5,
                                                    ),
                                                ],
                                                className="mb-4",
                                                align="center",
                                            ),

                                            html.Div(id="airquality-plot-container"),
                                        ],
                                        className="p-4"
                                    )
                                ],
                            ),
                            dbc.Tab(
                                label="Biodiversity",
                                tab_id="biodiversity",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("Biodiversity", className="mb-4"),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Dropdown(
                                                            id="bio-plot-dropdown",
                                                            options=[
                                                                {"label": "Monthly Observations", "value": "monthly_observations"},
                                                                {"label": "Map", "value": "map"},
                                                                {"label": "Fire Period Observations", "value": "fire_period_observations"},
                                                                {"label": "Description Analysis", "value": "description_analysis"},
                                                            ],
                                                            value="monthly_observations",
                                                            clearable=False,
                                                        ),
                                                        width=5,
                                                    ),

                                                    dbc.Col(
                                                        html.Div(
                                                            dbc.Checklist(
                                                                id="show-users-toggle",
                                                                options=[{"label": "Show users (right axis)", "value": "show_users"}],
                                                                value=[],
                                                                switch=True,
                                                                className="mb-0 big-toggle",
                                                            ),
                                                            id="show-users-wrapper",
                                                            style={"display": "flex", "justifyContent": "flex-end"},
                                                        ),
                                                        width=7,
                                                    ),
                                                ],
                                                className="mb-4",
                                                align="center",
                                            ),

                                            html.Div(id="bio-plot-container"),
                                        ],
                                        className="p-4"
                                    )
                                ],
                            ),
                            dbc.Tab(
                                label="News Reports",
                                tab_id="news",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("News Reports", className="mb-3"),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Dropdown(
                                                            id="news-plot-dropdown",
                                                            options=[
                                                                {"label": "Overview", "value": "news_overview"},
                                                                {"label": "Topic Modelling", "value": "news_topics"},
                                                                {"label": "Keyword Analysis", "value": "news_keywords"},
                                                            ],
                                                            value="news_overview",
                                                            clearable=False,
                                                        ),
                                                        width=5,
                                                    ),
                                                    dbc.Col(
                                                        html.Div(
                                                            dbc.Checklist(
                                                                id="show-news-quality-toggle",
                                                                options=[{"label": "Show extraction success (right axis)", "value": "show_quality"}],
                                                                value=["show_quality"],
                                                                switch=True,
                                                                className="mb-0 big-toggle",
                                                            ),
                                                            id="show-news-quality-wrapper",
                                                            style={"display": "flex", "justifyContent": "flex-end"},
                                                        ),
                                                        width=7,
                                                    ),
                                                ],
                                                className="mb-4",
                                                align="center",
                                            ),

                                            html.Div(id="news-plot-container"),
                                        ],
                                        className="p-4"
                                    )
                                ],
                            ),
                            dbc.Tab(
                                label="Cross-Correlation",
                                tab_id="crosscorrelation",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("Cross-Correlation", className="mb-3"),

                                            html.P(
                                                "Select two variables to examine how correlations change across daily time lags.",
                                                className="text-muted",
                                            ),

                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Driver variable"),
                                                            dcc.Dropdown(
                                                                id="ccf-var1-dropdown",
                                                                options=ccf_variable_options,
                                                                value="ppt",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),

                                                    dbc.Col(
                                                        [
                                                            html.Label("Lagged variable"),
                                                            dcc.Dropdown(
                                                                id="ccf-var2-dropdown",
                                                                options=ccf_variable_options,
                                                                value="pm25",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),

                                                    dbc.Col(
                                                        [
                                                            html.Label("Air quality station"),
                                                            dcc.Dropdown(
                                                                id="ccf-station-dropdown",
                                                                options=ccf_station_options,
                                                                value="average",
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        width=4,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),

                                            html.Div(id="ccf-plot-container"),
                                        ],
                                        className="p-4",
                                    )
                                ],
                            ),
                        ],
                        id="main-tabs",
                        active_tab="welcome",
                    )
                ]
            ),
            style={
                "backgroundColor": "var(--bs-primary)",
                "border": "2px solid black",
                "borderRadius": "18px",
                "padding": "12px",
            },
        ),
    ],
    fluid=True,
    className="p-4",
)


@app.callback(
    Output("bio-plot-container", "children"),
    Output("show-users-wrapper", "style"),
    Input("bio-plot-dropdown", "value"),
    Input("show-users-toggle", "value"),
)
def update_bio_plot_container(selected_plot, toggle_values):
    # show_users = "show_users" in toggle_values
    show_users = toggle_values is not None and "show_users" in toggle_values

    if selected_plot == "monthly_observations":
        content = html.Div(
            [
                dcc.Graph(
                    figure=fig_monthly_taxa_with_users(
                        taxon_counts=bio_data["taxon_counts"],
                        monthly_taxon=bio_data["monthly_taxon"],
                        user_counts=bio_data["user_counts"],
                        fire_date_str=bio_data["fire_date_str"],
                        show_users=show_users,
                    )
                )
            ]
        )
        toggle_style = {"display": "flex", "justifyContent": "flex-end"}

    elif selected_plot == "fire_period_observations":
        content = html.Div(
            [
                dcc.Graph(
                    figure=fig_taxa_counts(df_effort=df_effort)
                ),

                html.H5("Summary by Fire Period", className="mt-4 mb-3"),

                dash_table.DataTable(
                    data=period_summary.to_dict("records"),
                    columns=[
                        {"name": col, "id": col}
                        for col in period_summary.columns
                    ],
                    style_table={
                        "overflowX": "auto",
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "10px",
                        "fontFamily": "Arial",
                        "fontSize": "14px",
                        "whiteSpace": "normal",
                    },
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "#f8f9fa",
                        "border": "1px solid black",
                    },
                    style_data={
                        "border": "1px solid #ddd",
                    },
                ),
            ]
        )
        toggle_style = {"display": "none"}
    
    elif selected_plot == "map":
        content = html.Div(
            [
                html.Div(
                    dcc.Graph(
                        figure=fig_observation_map_slider(
                            df=df,
                            fire_start=bio_data["fire_start"]
                        )
                    ),
                    className="map-graph-wrapper"
                )
            ]
        )
        toggle_style = {"display": "none"}
    
    elif selected_plot == "description_analysis":
        content = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Top Bigrams in Observation Descriptions", className="mb-3"),

                                dash_table.DataTable(
                                    data=bigram_table.to_dict("records"),
                                    columns=[
                                        {"name": col, "id": col}
                                        for col in bigram_table.columns
                                    ],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "8px",
                                        "fontFamily": "Arial",
                                        "fontSize": "13px",
                                        "whiteSpace": "normal",
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#f8f9fa",
                                        "border": "1px solid black",
                                    },
                                    style_data={
                                        "border": "1px solid #ddd",
                                    },
                                ),

                                html.P(
                                    "Descriptions are available for only a subset of observations, so this view is exploratory.",
                                    className="mt-3 text-muted",
                                ),
                            ],
                            width=3,
                        ),

                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(
                                        figure=fig_description_counts_total(df),
                                        config={"displayModeBar": False},
                                    ),
                                    className="mb-4",
                                ),
                                html.Div(
                                    dcc.Graph(
                                        figure=fig_description_counts_time(
                                            df,
                                            bio_data["fire_date_str"]
                                        ),
                                        config={"displayModeBar": False},
                                    )
                                ),
                            ],
                            width=9,
                        ),
                    ],
                    className="g-4",
                )
            ]
        )
        toggle_style = {"display": "none"}

    else:
        content = html.Div("No plot selected.")
        toggle_style = {"display": "none"}

    return content, toggle_style

@app.callback(
    Output("news-plot-container", "children"),
    Output("show-news-quality-wrapper", "style"),
    Input("news-plot-dropdown", "value"),
    Input("show-news-quality-toggle", "value"),
)
def update_news_plot_container(selected_plot, toggle_values):
    show_quality = toggle_values is not None and "show_quality" in toggle_values

    if selected_plot == "news_overview":
        content = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Top 15 outlets", className="mb-3"),
                                dash_table.DataTable(
                                    data=top_outlets_table.to_dict("records"),
                                    columns=[
                                        {"name": col, "id": col}
                                        for col in top_outlets_table.columns
                                    ],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "8px",
                                        "fontFamily": "Arial",
                                        "fontSize": "13px",
                                        "whiteSpace": "normal",
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#f8f9fa",
                                        "border": "1px solid black",
                                    },
                                    style_data={
                                        "border": "1px solid #ddd",
                                    },
                                ),
                            ],
                            width=3,
                        ),

                        dbc.Col(
                            dcc.Graph(
                                figure=monthly_news_coverage_with_quality(
                                    report_df,
                                    show_quality=show_quality,
                                ),
                                config={"displayModeBar": False},
                            ),
                            width=9,
                        ),
                    ],
                    className="g-4",
                ),
            ]
        )

        toggle_style = {"display": "flex", "justifyContent": "flex-end"}

    elif selected_plot == "news_topics":
        content = html.Div(
            [
                dcc.Graph(
                    figure=topic_modelling(),
                ),

                html.H5("LLM Topic Labels and Descriptions", className="mt-4 mb-3"),

                dash_table.DataTable(
                    data=topic_llm_table.to_dict("records"),
                    columns=[
                        {"name": col, "id": col}
                        for col in topic_llm_table.columns
                    ],
                    style_table={
                        "overflowX": "auto",
                    },
                    style_cell={
                        "textAlign": "left",
                        "padding": "10px",
                        "fontFamily": "Arial",
                        "fontSize": "13px",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "#f8f9fa",
                        "border": "1px solid black",
                    },
                    style_data={
                        "border": "1px solid #ddd",
                    },
                    page_size=6,
                ),
            ]
        )

        toggle_style = {"display": "none"}

    elif selected_plot == "news_keywords":
        content = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Theme Keyword Groups", className="mb-3"),

                                dash_table.DataTable(
                                    data=theme_terms_table.to_dict("records"),
                                    columns=[
                                        {"name": col, "id": col}
                                        for col in theme_terms_table.columns
                                    ],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "8px",
                                        "fontFamily": "Arial",
                                        "fontSize": "13px",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#f8f9fa",
                                        "border": "1px solid black",
                                    },
                                    style_data={
                                        "border": "1px solid #ddd",
                                    },
                                ),
                            ],
                            width=3,
                            className="pt-5"
                        ),

                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(
                                        figure=themes_lines(theme_long),
                                        config={"displayModeBar": False},
                                    ),
                                    className="mb-4",
                                ),

                                html.Div(
                                    dcc.Graph(
                                        figure=themes_heatmap(theme_long),
                                        config={"displayModeBar": False},
                                    ),
                                ),
                            ],
                            width=9,
                        ),
                    ],
                    className="g-4",
                )
            ]
        )

        toggle_style = {"display": "none"}

    else:
        content = html.Div("No plot selected.")
        toggle_style = {"display": "none"}

    return content, toggle_style


@app.callback(
    Output("airquality-plot-container", "children"),
    Input("airquality-plot-dropdown", "value"),
)
def update_airquality_plot_container(selected_plot):
    if selected_plot == "monitor_map":
        content = html.Div(
            [
                dcc.Graph(
                    figure=build_map_figure(
                        perimeter_path="helpers/data/airquality/eaton_fire_extent.geojson",
                        monitors_path="helpers/data/airquality/site_names.csv",
                    ),
                    style={"width": "100%", "height": "650px"},
                    config={"responsive": True},
                )
            ],
            style={
                "width": "100%",
                "display": "flex",
                "justifyContent": "center",
            }
        )
    elif selected_plot == "historical_comparison":
        content = html.Div(
            [
                dcc.Graph(
                    figure=build_monthly_figure(
                        monthly_csv="helpers/data/airquality/monthly_baseline_oct_mar_2000_2024.csv",
                        event_csv="helpers/data/airquality/event_window_daily.csv",
                    ),
                    style={"width": "90%", "height": "620px"},
                    config={"responsive": True, "displayModeBar": False},
                )
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "width": "100%",
            },
        )

    elif selected_plot == "focal_stations":
        content = html.Div(
            [
                dcc.Graph(
                    figure=build_focal_figure(
                        event_csv="helpers/data/airquality/event_window_daily.csv",
                        site_csv="helpers/data/airquality/event_window_site_specific.csv",
                        monthly_csv="helpers/data/airquality/monthly_baseline_oct_mar_2000_2024.csv",
                    ),
                    style={"width": "90%", "height": "620px"},
                    config={"responsive": True, "displayModeBar": False},
                )
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "width": "100%",
            },
        )

    else:
        content = html.Div("No plot selected.")

    return content


@app.callback(
    Output("ccf-plot-container", "children"),
    Input("ccf-var1-dropdown", "value"),
    Input("ccf-var2-dropdown", "value"),
    Input("ccf-station-dropdown", "value"),
)
def update_ccf_plot(var1, var2, station_value):
    station = None

    if station_value != "average":
        station = station_value

    fig = cross_correlation(
        var1=var1,
        var2=var2,
        station=station,
    )

    return html.Div(
        dcc.Graph(
            figure=fig,
            config={"displayModeBar": False},
            style={"width": "90%", "height": "650px"},
        ),
        style={
            "display": "flex",
            "justifyContent": "center",
            "width": "100%",
        },
    )


if __name__ == "__main__":
    app.run(debug=True)

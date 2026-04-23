from dash import Dash, html, dcc, Input, Output, dash_table
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from helpers.biodiversity import (
    fig_monthly_taxa_with_users,
    load_biodiversity_data,
    period_counts,
    fig_taxa_counts,
    clean_data
)

BRITE = "https://bootswatch.com/5/brite/bootstrap.min.css"

app = Dash(__name__, external_stylesheets=[BRITE])
bio_data = load_biodiversity_data()
df = clean_data(bio_data['df'])
df_effort, period_summary = period_counts(df)
period_summary["Observations Per User"] = period_summary["Observations Per User"].round(2)
period_summary["Share of Total Observations"] = period_summary["Share of Total Observations"].round(1)
period_summary["Share of Total Users"] = period_summary["Share of Total Users"].round(1)

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
                                label="Overall",
                                tab_id="overall",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("Overall", className="mb-3"),
                                            html.P("Put your overall dashboard summary, key stats, and combined plots here."),
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
                                                                {"label": "Fire Period Observations", "value": "fire_period_observations"},
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
                                            html.P("Put your article topic analysis, environmental themes, and news summaries here."),
                                        ],
                                        className="p-4"
                                    )
                                ],
                            ),
                        ],
                        id="main-tabs",
                        active_tab="overall",
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

# @app.callback(
#     Output("biodiversity-monthly-graph", "figure"),
#     Input("show-users-toggle", "value")
# )
# def update_biodiversity_graph(toggle_values):
#     show_users = "show_users" in toggle_values

#     fig = fig_monthly_taxa_with_users(
#         taxon_counts=bio_data["taxon_counts"],
#         monthly_taxon=bio_data["monthly_taxon"],
#         user_counts=bio_data["user_counts"],
#         fire_date_str=bio_data["fire_date_str"],
#         show_users=show_users
#     )

#     return fig

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

    else:
        content = html.Div("No plot selected.")
        toggle_style = {"display": "none"}

    return content, toggle_style

if __name__ == "__main__":
    app.run(debug=True)
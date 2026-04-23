import re
import os
import time
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from shapely.geometry import Point
from datetime import datetime
from pyinaturalist import *
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

from pathlib import Path
import pandas as pd

fire_start = pd.Timestamp("2025-01-07")
fire_end = pd.Timestamp("2025-01-31")
fire_date_str = "2025-01-07"


def load_biodiversity_data():
    base_path = Path("helpers/data/biodiversity")

    file_map = {
        "df": "df_inat_pasalt.csv",
        "monthly_counts": "monthly_counts.csv",
        "weekly_counts": "weekly_counts.csv",
        "taxon_counts": "taxon_counts.csv",
        "monthly_taxon": "monthly_taxon.csv",
        "user_counts": "user_counts.csv",
        "top_users": "top_users.csv",
        "missingness": "missingness.csv",
    }

    data = {}

    for key, filename in file_map.items():
        full_path = base_path / filename
        data[key] = pd.read_csv(full_path)

    data["fire_start"] = pd.Timestamp("2025-01-07")
    data["fire_end"] = pd.Timestamp("2025-01-31")
    data["fire_date_str"] = "2025-01-07"

    return data

def clean_data(df):
    df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["id", "observed_on", "latitude", "longitude"])
    df = df.drop_duplicates(subset="id")
    df["has_taxon"] = df["taxon_id"].notna()
    df["has_description"] = df["description"].fillna("").str.strip().ne("")
    df["year"] = df["observed_on"].dt.year
    df["month"] = df["observed_on"].dt.to_period("M").astype(str)
    df["week"] = df["observed_on"].dt.to_period("W").astype(str)
    df["day"] = df["observed_on"].dt.date
    df["period"] = np.where(
        df["observed_on"] < fire_start, "pre-fire",
        np.where(df["observed_on"] <= fire_end, "fire-period", "post-fire")
    )
    return df

def fig_monthly_taxa(taxon_counts, monthly_taxon, fire_date_str):
    top_taxa_names = taxon_counts.head(6)["iconic_taxon_name"].dropna().tolist()
    monthly_taxon_top = monthly_taxon[
        monthly_taxon["iconic_taxon_name"].isin(top_taxa_names)
    ].copy()
    fig_monthly_taxa = px.area(
        monthly_taxon_top,
        x="month",
        y="n_obs",
        color="iconic_taxon_name",
        title="Monthly Observations by Broad Taxonomic Group",
        labels={
            "month_dt": "Month",
            "n_obs": "Number of observations",
            "iconic_taxon_name": "Iconic taxon"
        }
    )
    fig_monthly_taxa.add_vline(
        x=fire_date_str,
        line_dash="dash",
        line_width=1
    )
    fig_monthly_taxa.add_annotation(
        x=fire_date_str,
        y=monthly_taxon_top["n_obs"].max(),
        text="Eaton Fire starts",
        showarrow=True,
        yshift=10
    )
    fig_monthly_taxa.update_layout(
        template="plotly_white",
        # width=950,
        height=550
    )
    return fig_monthly_taxa

def period_counts(df):
    df_effort = df.copy()
    df_effort["period_3"] = np.where(
        df_effort["observed_on"] < fire_start,
        "Pre-fire",
        np.where(df_effort["observed_on"] <= fire_end, "Fire period", "Post-fire")
    )

    period_summary = (
        df_effort.groupby("period_3")
        .agg(
            n_obs=("id", "count"),
            n_unique_users=("user_id", "nunique"),
            n_descriptions=("has_description", "sum")
        )
        .reset_index()
    )

    period_summary["obs_per_user"] = period_summary["n_obs"] / period_summary["n_unique_users"]
    # period_summary["pct_with_description"] = (
    #     period_summary["n_descriptions"] / period_summary["n_obs"] * 100
    # )
    period_summary["share_of_total_obs"] = period_summary["n_obs"] / period_summary["n_obs"].sum() * 100
    period_summary["share_of_total_users"] = period_summary["n_unique_users"] / period_summary["n_unique_users"].sum() * 100
    period_summary.columns = ['Period', '# of Observations', '# of Unique Users', '# of Descriptions', 'Observations Per User', 'Share of Total Observations', 'Share of Total Users']
    return df_effort, period_summary

def fig_taxa_counts(df_effort):
    taxon_user_period = (
        df_effort.groupby(["period_3", "iconic_taxon_name"])
        .agg(
            n_obs=("id", "count"),
            n_users=("user_id", "nunique")
        )
        .reset_index()
    )
    taxon_user_period["obs_per_user"] = taxon_user_period["n_obs"] / taxon_user_period["n_users"]
    taxon_user_period
    taxon_user_period = (
        df_effort.groupby(["period_3", "iconic_taxon_name"])
        .agg(
            n_obs=("id", "count"),
            n_users=("user_id", "nunique")
        )
        .reset_index()
    )

    top_taxa = (
        df_effort.groupby("iconic_taxon_name")
        .size()
        .sort_values(ascending=False)
        .head(6)
        .index
        .tolist()
    )

    taxon_user_period_top = taxon_user_period[
        taxon_user_period["iconic_taxon_name"].isin(top_taxa)
    ].copy()

    taxon_user_period_top["obs_per_user"] = (
        taxon_user_period_top["n_obs"] / taxon_user_period_top["n_users"]
    )
    fig_taxon_per_user = px.bar(
        taxon_user_period_top,
        x="iconic_taxon_name",
        y="obs_per_user",
        color="period_3",
        barmode="group",
        title="Taxonomic Observations per User by Fire Period",
        labels={
            "iconic_taxon_name": "Taxonomic group",
            "obs_per_user": "Observations per user",
            "period_3": "Period"
        }
    )

    fig_taxon_per_user.update_layout(
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title_text=""
        ),
        margin=dict(r=40, b=90)
    )

    return fig_taxon_per_user


def fig_monthly_taxa_with_users(
    taxon_counts,
    monthly_taxon,
    user_counts,
    fire_date_str,
    show_users=False
):
    top_taxa_names = taxon_counts.head(6)["iconic_taxon_name"].dropna().tolist()

    monthly_taxon_top = monthly_taxon[
        monthly_taxon["iconic_taxon_name"].isin(top_taxa_names)
    ].copy()

    monthly_taxon_top["month"] = pd.to_datetime(monthly_taxon_top["month"])
    user_counts_plot = user_counts.copy()
    user_counts_plot["month"] = pd.to_datetime(user_counts_plot["month"])

    fig = px.area(
        monthly_taxon_top,
        x="month",
        y="n_obs",
        color="iconic_taxon_name",
        title="Monthly Observations by Broad Taxonomic Group",
        labels={
            "month": "Month",
            "n_obs": "Number of observations",
            "iconic_taxon_name": "Iconic taxon"
        }
    )

    fig.add_vline(
        x=fire_date_str,
        line_dash="dash",
        line_width=1
    )

    fig.add_annotation(
        x=fire_date_str,
        y=monthly_taxon_top["n_obs"].max(),
        text="Eaton Fire starts",
        showarrow=True,
        yshift=10
    )

    if show_users:
        fig.add_trace(
            go.Scatter(
                x=user_counts_plot["month"],
                y=user_counts_plot["n_unique_users"],
                mode="lines+markers",
                name="Unique users",
                yaxis="y2",
                line=dict(width=3, dash="dot"),
                showlegend=False
            )
        )

        fig.update_layout(
            yaxis2=dict(
                title="Number of unique users",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    # fig.update_layout(
    #     template="plotly_white",
    #     height=550,
    #     legend_title_text="Iconic taxon",
    #     hovermode="x unified"
    # )
    fig.update_layout(
        template="plotly_white",
        height=550,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title_text=""
        ),
        margin=dict(r=80, b=100)
    )

    return fig

def fig_description_counts_time(df):
    df_text = df[df["has_description"]].copy()
    df_text["month_dt"] = pd.to_datetime(df_text["month"])
    focus_terms = ["fire", "ash", "burn", "trail", "coyote"]
    keyword_time_rows = []

    for term in focus_terms:
        temp = (
            df_text.assign(has_term=df_text["description"].str.contains(
                rf"\b{term}\b", case=False, na=False, regex=True
            ))
            .groupby("month_dt")["has_term"]
            .sum()
            .reset_index(name="count")
        )
        temp["term"] = term
        keyword_time_rows.append(temp)

    keyword_time = pd.concat(keyword_time_rows, ignore_index=True)

    fig_keyword_time = px.line(
        keyword_time,
        x="month_dt",
        y="count",
        color="term",
        markers=True,
        title="Selected Keywords in Descriptions Over Time",
        labels={"month_dt": "Month", "count": "Count of descriptions"}
    )

    fig_keyword_time.add_vline(x=fire_date_str, line_dash="dash", line_width=1)

    fig_keyword_time.update_layout(
        template="plotly_white",
        width=1000,
        height=500
    )

    return fig_keyword_time


def main():
    df = pd.read_csv('./data/df_inat_pasalt.csv')
    monthly_counts = pd.read_csv("data/monthly_counts.csv")
    weekly_counts = pd.read_csv("data/weekly_counts.csv")
    taxon_counts = pd.read_csv("data/taxon_counts.csv")
    monthly_taxon = pd.read_csv("data/monthly_taxon.csv")
    user_counts = pd.read_csv("data/user_counts.csv")
    top_users = pd.read_csv("data/top_users.csv")
    missingness = pd.read_csv("data/missingness.csv")

if __name__ == "__main__":
    main()
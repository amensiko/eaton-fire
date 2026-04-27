import os
import time
import re
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import date

def load_and_clean_news_data():
    fixed_df = pd.read_csv("helpers/data/news/all_stories_fixed.csv")
    news_df = fixed_df[fixed_df["text_len"] >= 1000].copy()
    news_df = news_df[news_df["language"] == "en"].copy()
    report_df = fixed_df.copy()
    report_df["publish_date"] = pd.to_datetime(report_df["publish_date"], errors="coerce")
    report_df["indexed_date"] = pd.to_datetime(report_df["indexed_date"], errors="coerce")
    report_df["text_len"] = pd.to_numeric(report_df["text_len"], errors="coerce").fillna(0)
    report_df["text"] = report_df["text"].fillna("")
    report_df["has_text"] = report_df["text_len"] > 0
    report_df["usable_text"] = report_df["text_len"] >= 1000

    return fixed_df, news_df, report_df


def daily_article_counts(report_df):
    daily_counts = (
        report_df.dropna(subset=["publish_date"])
        .groupby(report_df["publish_date"].dt.date)
        .size()
        .reset_index(name="n_articles")
    )

    daily_counts["publish_date"] = pd.to_datetime(daily_counts["publish_date"])

    fig_daily = px.line(
        daily_counts,
        x="publish_date",
        y="n_articles",
        title="Daily Eaton Fire article counts",
        labels={"publish_date": "Publish date", "n_articles": "Number of articles"},
    )

    fig_daily.update_layout(
        template="plotly_white",
        # width=1100,
        # height=500,
    )
    # fig_daily.update_traces(line_color="purple")

    return fig_daily

def top_outlets_counts(report_df):
    top_outlets = (
        report_df["media_name"]
        .fillna("Unknown")
        .value_counts()
        .head(15)
        .reset_index()
    )

    top_outlets.columns = ["media_name", "n_articles"]

    # Reverse for nicer horizontal ordering
    top_outlets = top_outlets.sort_values("n_articles", ascending=True)

    fig_outlets = px.bar(
        top_outlets,
        x="n_articles",
        y="media_name",
        orientation="h",
        title="Top 15 outlets in the Eaton Fire news dataset",
        labels={"media_name": "Outlet", "n_articles": "Number of articles"},
        color_discrete_sequence=['pink'],
    )

    fig_outlets.update_layout(
        template="plotly_white",
        # width=1000,
        # height=650
    )

    return fig_outlets


def monthly_text_extraction_success(report_df):
    monthly_quality = (
        report_df.dropna(subset=["publish_date"])
        .assign(month=report_df["publish_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            total_articles=("mc_id", "count"),
            with_text=("has_text", "sum"),
            usable_text=("usable_text", "sum"),
            median_text_len=("text_len", "median")
        )
        .reset_index()
    )

    monthly_quality["pct_with_text"] = 100 * monthly_quality["with_text"] / monthly_quality["total_articles"]
    monthly_quality["pct_usable_text"] = 100 * monthly_quality["usable_text"] / monthly_quality["total_articles"]

    monthly_quality_long = monthly_quality.melt(
        id_vars="month",
        value_vars=["pct_with_text", "pct_usable_text"],
        var_name="metric",
        value_name="percent"
    )

    monthly_quality_long["metric"] = monthly_quality_long["metric"].replace({
        "pct_with_text": "Has text",
        "pct_usable_text": ">=1000 chars"
    })

    fig_quality = px.line(
        monthly_quality_long,
        x="month",
        y="percent",
        color="metric",
        markers=True,
        title="Monthly text extraction success",
        labels={"month": "Month", "percent": "Percent of articles", "metric": ""}
    )

    fig_quality.update_layout(
        template="plotly_white",
        # width=1000,
        # height=500
    )

    return fig_quality

def article_size_hist(report_df):
    zoom_df = report_df[report_df["text_len"] <= 15000].copy()

    fig_text_len_zoom = px.histogram(
        zoom_df,
        x="text_len",
        nbins=50,
        title="Distribution of extracted article text lengths (<= 15,000 chars)",
        labels={"text_len": "Extracted text length (characters)", "count": "Number of articles"},
        color_discrete_sequence=['green']
    )

    fig_text_len_zoom.update_layout(
        template="plotly_white",
        # width=1000,
        # height=500,
    )

    return fig_text_len_zoom


def monthly_news_coverage_with_quality(monthly, show_quality=False):
    monthly = monthly.copy()
    monthly["month_dt"] = pd.to_datetime(monthly["month_dt"])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=monthly["month_dt"],
            y=monthly["total_articles"],
            name="Monthly articles",
        )
    )

    if show_quality:
        fig.add_trace(
            go.Scatter(
                x=monthly["month_dt"],
                y=monthly["pct_with_text"],
                mode="lines+markers",
                name="Has text (%)",
                yaxis="y2",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly["month_dt"],
                y=monthly["pct_usable_text"],
                mode="lines+markers",
                name="≥1000 chars (%)",
                yaxis="y2",
            )
        )

        fig.update_layout(
            yaxis2=dict(
                title="Text extraction success (%)",
                overlaying="y",
                side="right",
                range=[0, 100],
                showgrid=False,
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="Monthly Eaton Fire News Coverage",
        height=550,
        margin=dict(l=40, r=80, t=80, b=90),
        yaxis_title="Number of articles",
        xaxis_title="Month",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title_text=""
        ),
    )

    return fig


def topic_modelling():
    plot_df = pd.read_csv("helpers/data/news/llama_plot_df_slim.csv")
    # plot_df = pd.read_csv("helpers/data/news/llama_plot_df.csv")
    fig_phase = px.scatter(
        plot_df,
        # plot_df.sample(min(5000, len(plot_df)), random_state=42),
        x="x",
        y="y",
        color="topic_label_clean",
        hover_data=["title", "media_name", "phase", "month", "topic"],
        animation_frame="phase",
            category_orders={
            "phase": ["active_fire", "early_recovery", "later_recovery"]
        },
        title="BERTopic clusters across timeline phases",
        opacity=0.7
    )

    fig_phase.update_layout(
        template="plotly_white",
        height=700,
        legend_title_text="Topic Labels"
    )

    return fig_phase

def topic_modelling_llm_table():
    topic_description_table = pd.read_csv("helpers/data/news/topic_description_table.csv")
    return topic_description_table


def themes_lines(theme_long):
    theme_month_counts = (
        theme_long.groupby(["month", "theme"])
        .size()
        .reset_index(name="n_articles")
    )

    fig_theme_month = px.line(
        theme_month_counts,
        x="month",
        y="n_articles",
        color="theme",
        markers=True,
        title="Environmental theme prevalence over time",
        labels={"month": "Month", "n_articles": "Number of articles", "theme": "Theme"}
    )

    fig_theme_month.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=70, b=60),
    )

    return fig_theme_month

def themes_heatmap(theme_long):
    theme_phase_heatmap = (
        theme_long.groupby(["phase", "theme"])
        .size()
        .reset_index(name="n_articles")
        .pivot(index="phase", columns="theme", values="n_articles")
        .fillna(0)
    )

    fig_theme_heatmap = px.imshow(
        theme_phase_heatmap,
        text_auto=True,
        aspect="auto",
        title="Environmental themes across phases",
        labels={"x": "Theme", "y": "Phase", "color": "Articles"}
    )

    fig_theme_heatmap.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=70, b=60),
    )

    return fig_theme_heatmap

def themes_keywords():
    theme_dict = {
        "Smoke / Air Quality": [
            "air quality", "wildfire smoke", "smoke", "smoke exposure", "pm2.5",
            "particulate", "pollution", "soot"
        ],
        "Debris / Cleanup": [
            "debris", "debris removal", "cleanup", "hazardous debris",
            "ash", "ash cleanup", "waste"
        ],
        "Water / Runoff / Contamination": [
            "water quality", "drinking water", "contamination", "contaminated",
            "runoff", "ash runoff", "watershed", "sediment", "erosion", "toxic", "toxins"
        ],
        "Utilities / Fire Cause": [
            "edison", "sce", "utility", "utilities", "power lines",
            "transmission", "equipment", "fire cause", "ignition"
        ],
        "Wildlife / Habitat / Trees": [
            "wildlife", "habitat", "ecosystem", "vegetation", "tree", "trees",
            "animals", "species", "biodiversity"
        ],
        "Public Health / Recovery": [
            "public health", "respiratory", "health", "recovery", "rebuild",
            "rebuilding", "displacement", "mental health"
        ]
    }
    theme_terms_table = pd.DataFrame(
        [
            {
                "Theme": theme,
                "Keywords": ", ".join(keywords)
            }
            for theme, keywords in theme_dict.items()
        ]
    )
    return theme_terms_table
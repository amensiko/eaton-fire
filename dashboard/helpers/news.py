import os
import time
import re
import umap
import torch
import requests
import pandas as pd
import numpy as np
import mediacloud.api
import trafilatura
import plotly.express as px
import plotly.graph_objects as go

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from tqdm.auto import tqdm
from datetime import date
from tenacity import retry, stop_after_attempt, wait_exponential
from readability.readability import Document
from bs4 import BeautifulSoup
from newspaper import fulltext
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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


def monthly_news_coverage_with_quality(report_df, show_quality=False):
    monthly = (
        report_df.dropna(subset=["publish_date"])
        .assign(month=report_df["publish_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            total_articles=("mc_id", "count"),
            with_text=("has_text", "sum"),
            usable_text=("usable_text", "sum"),
        )
        .reset_index()
    )

    monthly["month_dt"] = pd.to_datetime(monthly["month"])
    monthly["pct_with_text"] = 100 * monthly["with_text"] / monthly["total_articles"]
    monthly["pct_usable_text"] = 100 * monthly["usable_text"] / monthly["total_articles"]

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

def clean_text_for_topic_modeling(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # remove line breaks
    text = re.sub(r"[\r\n\t]+", " ", text)
    
    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in all_stopwords and len(tok) > 2]
    
    return " ".join(tokens)

def assign_phase(dt):
    if pd.isna(dt):
        return None
    elif dt <= pd.Timestamp("2025-01-31"):
        return "active_fire"
    elif dt <= pd.Timestamp("2025-03-31"):
        return "early_recovery"
    else:
        return "later_recovery"

def prepare_text(news_df):
    model_df = news_df.copy().reset_index(drop=True)
    model_df["publish_date"] = pd.to_datetime(model_df["publish_date"], errors="coerce")
    model_df["title_norm"] = (
        model_df["title"]
        .fillna("")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    model_df["text_norm"] = (
        model_df["text"]
        .fillna("")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    model_df = model_df.drop_duplicates(subset=["clean_url"])
    model_df = model_df.drop_duplicates(subset=["title_norm"])
    # model_df = model_df.drop_duplicates(subset=["text_norm"])

    model_df["phase"] = model_df["publish_date"].apply(assign_phase)
    model_df["month"] = model_df["publish_date"].dt.to_period("M").astype(str)
    model_df["phase"].value_counts(dropna=False)
    model_df["env_relevant"] = (
        model_df["title"].fillna("").str.lower().str.contains(env_pattern, regex=True) |
        model_df["text"].fillna("").str.lower().str.contains(env_pattern, regex=True)
    )
    env_df = model_df[model_df['env_relevant'] == True].reset_index(drop=True)


    final_model_df = model_df[[
        "mc_id", "title", "media_name", "publish_date", "month", "phase",
        "url", "clean_url", "text", "text_len", "env_relevant"
    ]].copy()
    final_model_df["text_clean"] = final_model_df["text"].apply(clean_text_for_topic_modeling)
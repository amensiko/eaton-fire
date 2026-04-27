import pandas as pd
from pathlib import Path

DATA = Path("helpers/data/news")

# Load big source once
report_df = pd.read_csv(DATA / "all_stories_fixed.csv")

report_df["publish_date"] = pd.to_datetime(report_df["publish_date"], errors="coerce")
report_df["text_len"] = pd.to_numeric(report_df["text_len"], errors="coerce").fillna(0)
report_df["has_text"] = report_df["text_len"] > 0
report_df["usable_text"] = report_df["text_len"] >= 1000

# Monthly coverage + text quality
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

monthly.to_csv(DATA / "news_monthly_coverage_quality.csv", index=False)

# Top outlets table
top_outlets = (
    report_df["media_name"]
    .fillna("Unknown")
    .value_counts()
    .head(15)
    .reset_index()
)

top_outlets.columns = ["Outlet", "Number of articles"]
top_outlets.to_csv(DATA / "news_top_outlets_table.csv", index=False)

# Slim topic plot file
plot_df = pd.read_csv(DATA / "llama_plot_df.csv")

keep_cols = [
    "x",
    "y",
    "topic_label_clean",
    "title",
    "media_name",
    "phase",
    "month",
    "topic",
]

plot_df_slim = plot_df[keep_cols].copy()

# plot_df_slim = (
#     plot_df_slim
#     .groupby("phase", group_keys=False)
#     .apply(lambda x: x.sample(min(len(x), 5000), random_state=42))
# )

plot_df_slim.to_csv(DATA / "llama_plot_df_slim.csv", index=False)

print("Done creating small news dashboard files.")
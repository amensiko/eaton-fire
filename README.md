# 🔥 Eaton Fire Analysis

Environmental Impacts of the Eaton Fire using Air Quality, Biodiversity, Weather, and News Data.

👉 **Live Dashboard:** **[View Interactive Dashboard](https://f6888fec-30af-49ce-a6eb-071e42d7dfad.plotly.app/)**

---

## Overview

This project analyzes the environmental and societal impacts of the Eaton Fire by integrating multiple data sources:

- 🌫️ Air Quality (PM2.5 and station-level trends)
- 🌱 Biodiversity (iNaturalist observations and activity patterns)
- 🌦️ Weather (historical trends and anomaly detection)
- 📰 News Coverage (volume, themes, and temporal dynamics)

The goal is to explore how these systems interact before, during, and after the fire.

---

## Dashboard

The interactive dashboard presents the final results of the analysis across several sections:

- **Air Quality** – Station map, historical baselines, and time series  
- **Biodiversity** – Observation trends and user activity  
- **Weather** – Long-term trends and anomaly comparisons  
- **News Reports** – Article volume, topic structure, and coverage patterns  
- **Cross-Correlation** – Relationships between variables across time lags  

All visualizations are built using Plotly Dash.

---

## Key Features

- 📊 Multi-source data integration (environmental + social signals)  
- ⏱️ Temporal analysis (daily + monthly aggregation)  
- 🔄 Cross-correlation analysis with lag structure  
- 📉 Trend detection (Mann-Kendall tests for weather)  
- 🧠 Topic modeling + news summarization  
- 🗺️ Spatial visualization (air quality stations + fire extent)  

---

## Data Processing

To make deployment feasible, large raw datasets were reduced into smaller, precomputed files used by the dashboard.

**Examples:**
- `all_stories.csv` → aggregated into lightweight summary tables  
- Weather and biodiversity data → pre-aggregated to monthly/daily formats  

Scripts for preprocessing are included in the repository.

---

## Setup (Local)

```bash
pip install -r dashboard/requirements.txt
cd dashboard
python app.py
```

---

## Deployment

The dashboard is deployed using **Plotly Cloud**.

- Uses a lightweight dataset (<100MB)  
- Configured via `plotly-cloud.toml`  
- Public access enabled via link  

---

## Contributors

- Anastasia Menshikova 
- Will Behm 
- Erik Lagerquist  

---

## Notes

- This repository includes both **analysis code** and **final dashboard outputs**  
- The `dashboard/` folder contains the production-ready app  
- Other folders contain exploratory work, preprocessing, and experiments  
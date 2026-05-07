"""
Eaton Fire Air Quality Analysis
Pulls PM2.5 data from EPA AQS API for LA County (CA):
  - Historical baseline: 2000–2024, county-wide daily FRM (88101)
  - Event window: Jan–Mar 2025, per-site hourly non-FRM (88502) where
    available, daily FRM (88101) for sites without a continuous monitor
"""

import calendar
import os
import time
import requests
import pandas as pd

# ── Credentials ───────────────────────────────────────────────────────────────
# Set these or use environment variables:
#   export AQS_EMAIL="your@email.com"
#   export AQS_KEY="your_api_key"

AQS_EMAIL = os.getenv("AQS_EMAIL", "elagerqu@uvm.edu")
AQS_KEY   = os.getenv("AQS_KEY",   "taupebird28")

# ── Geography ─────────────────────────────────────────────────────────────────
STATE  = "06"   # California
COUNTY = "037"  # Los Angeles County

# ── Run toggles ───────────────────────────────────────────────────────────────
PULL_HISTORICAL = False   # 2000–2024 county-wide daily FRM  (slow — ~25 API calls)
PULL_EVENT      = True   # Jan–Mar 2025 per-site            (moderate — 1 call per site)

# ── Parameters ────────────────────────────────────────────────────────────────
# Historical pull uses FRM only — long, consistent county-wide record
PARAM_HISTORICAL = "88101"   # PM2.5 FRM — used for baseline

# Event pull prefers continuous non-FRM (hourly); falls back to FRM (daily)
# for sites that don't operate a continuous monitor
PARAM_NONFRM = "88502"       # PM2.5 non-FRM continuous — hourly, event pull
PARAM_FRM    = "88101"       # PM2.5 FRM — daily fallback, event pull

BASE_URL = "https://aqs.epa.gov/data/api"


# ── API helpers ───────────────────────────────────────────────────────────────

def _check_response(data: dict, label: str) -> bool:
    """Return True if response is successful, else print warning and return False."""
    status = data.get("Header", [{}])[0].get("status")
    if status != "Success":
        msg = data.get("Header", [{}])[0].get("error", "Unknown error")
        print(f"  API warning ({label}): {msg}")
        return False
    return True


def get_daily_by_county(bdate: str, edate: str, param: str = PARAM_HISTORICAL) -> pd.DataFrame:
    """
    Fetch county-wide daily summary data between bdate and edate.
    Dates in YYYYMMDD format. Returns DataFrame or empty DataFrame on failure.
    Used for historical baseline pull.
    """
    resp = requests.get(f"{BASE_URL}/dailyData/byCounty", params={
        "email": AQS_EMAIL, "key": AQS_KEY,
        "param": param, "bdate": bdate, "edate": edate,
        "state": STATE, "county": COUNTY,
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not _check_response(data, f"{bdate}–{edate}"):
        return pd.DataFrame()
    rows = data.get("Data", [])
    if not rows:
        print(f"  No data returned for {bdate}–{edate}")
        return pd.DataFrame()
    return pd.DataFrame(rows)


def get_county_monitors(param: str, bdate: str, edate: str) -> tuple[list[str], pd.DataFrame]:
    """
    Return (site_number_list, monitor_metadata_df) for LA County monitors active
    during bdate–edate for the given param.
    """
    resp = requests.get(f"{BASE_URL}/monitors/byCounty", params={
        "email": AQS_EMAIL, "key": AQS_KEY,
        "param": param, "bdate": bdate, "edate": edate,
        "state": STATE, "county": COUNTY,
    }, timeout=30)
    resp.raise_for_status()
    rows = resp.json().get("Data", [])
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    sites = sorted({m["site_number"] for m in rows})
    return sites, df


def get_hourly_by_site(site: str, bdate: str, edate: str) -> pd.DataFrame:
    """
    Fetch hourly 88502 sample data for a single site.
    Returns DataFrame with a 'sample_measurement' value column.
    """
    resp = requests.get(f"{BASE_URL}/sampleData/bySite", params={
        "email": AQS_EMAIL, "key": AQS_KEY,
        "param": PARAM_NONFRM, "bdate": bdate, "edate": edate,
        "state": STATE, "county": COUNTY, "site": site,
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not _check_response(data, f"site {site} hourly"):
        return pd.DataFrame()
    rows = data.get("Data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["resolution"] = "hourly"
    df["param_code"] = PARAM_NONFRM
    return df


def get_daily_by_site(site: str, bdate: str, edate: str) -> pd.DataFrame:
    """
    Fetch daily 88101 summary data for a single site.
    Returns DataFrame with an 'arithmetic_mean' value column.
    """
    resp = requests.get(f"{BASE_URL}/dailyData/bySite", params={
        "email": AQS_EMAIL, "key": AQS_KEY,
        "param": PARAM_FRM, "bdate": bdate, "edate": edate,
        "state": STATE, "county": COUNTY, "site": site,
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not _check_response(data, f"site {site} daily"):
        return pd.DataFrame()
    rows = data.get("Data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["resolution"] = "daily"
    df["param_code"] = PARAM_FRM
    return df


def fetch_year(year: int) -> pd.DataFrame:
    """Fetch one full calendar year of county-wide daily FRM data."""
    bdate = f"{year}0101"
    edate = f"{year}1231"
    print(f"  Fetching {year}...")
    df = get_daily_by_county(bdate, edate)
    time.sleep(1.5)   # be polite to the API
    return df


# ── Main fetch: historical ────────────────────────────────────────────────────

def fetch_historical_data() -> pd.DataFrame:
    """Fetch 2000–2024 county-wide daily FRM (88101). One API call per year."""
    frames = []
    for year in range(2000, 2025):
        df = fetch_year(year)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No historical data retrieved.")
    return pd.concat(frames, ignore_index=True)


# ── Main fetch: event window ──────────────────────────────────────────────────

def fetch_event_data() -> pd.DataFrame:
    """
    Fetch Jan–Mar 2025 PM2.5 per site:
      - 88502 hourly for sites with a continuous monitor
      - 88101 daily for sites that only have FRM
    Also saves site_names.csv with one row per site.
    """
    bdate, edate = "20250101", "20250331"
    frames = []

    nonfrm_sites, nonfrm_meta = get_county_monitors(PARAM_NONFRM, bdate, edate)
    frm_sites,    frm_meta    = get_county_monitors(PARAM_FRM,    bdate, edate)
    frm_only = sorted(set(frm_sites) - set(nonfrm_sites))

    # Save one row per site with name/location metadata
    meta_cols = ["site_number", "local_site_name", "latitude", "longitude",
                 "city", "address", "cbsa"]
    all_meta = pd.concat([nonfrm_meta, frm_meta], ignore_index=True)
    if not all_meta.empty:
        keep = [c for c in meta_cols if c in all_meta.columns]
        site_names = (
            all_meta[keep]
            .drop_duplicates("site_number")
            .sort_values("site_number")
            .reset_index(drop=True)
        )
        site_names.to_csv("site_names.csv", index=False)
        print(f"  Saved site_names.csv ({len(site_names)} sites)")

    print(f"  88502 sites ({len(nonfrm_sites)}): {nonfrm_sites}")
    print(f"  88101-only sites ({len(frm_only)}): {frm_only}")

    for site in nonfrm_sites:
        print(f"  Fetching hourly 88502 — site {site}...")
        df = get_hourly_by_site(site, bdate, edate)
        if not df.empty:
            frames.append(df)
        time.sleep(1.0)

    for site in frm_only:
        print(f"  Fetching daily 88101 — site {site}...")
        df = get_daily_by_site(site, bdate, edate)
        if not df.empty:
            frames.append(df)
        time.sleep(1.0)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Processing ────────────────────────────────────────────────────────────────

FIRE_SEASON_MONTHS = [10, 11, 12, 1, 2, 3]  # October–March


def process_historical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From county-wide daily FRM data, compute:
      - month_climatology : single clim_mean/std per month across all baseline years
      - monthly_baseline  : per season-year per month mean/std/count/max
    Both cover Oct–Mar fire season only, 2000–2024.
    """
    df = df.copy()
    df["date_local"]      = pd.to_datetime(df["date_local"])
    df["arithmetic_mean"] = pd.to_numeric(df["arithmetic_mean"], errors="coerce")
    df = df.dropna(subset=["arithmetic_mean"])
    df["year"]  = df["date_local"].dt.year
    df["month"] = df["date_local"].dt.month

    baseline = df[
        (df["year"] >= 2000) & (df["year"] <= 2024) &
        df["month"].isin(FIRE_SEASON_MONTHS)
    ].copy()

    # Oct–Dec of year Y belongs to season Y+1 (same season as Jan–Mar Y+1)
    baseline["season_year"] = baseline.apply(
        lambda r: r["year"] + 1 if r["month"] >= 10 else r["year"], axis=1
    )

    monthly_baseline = (
        baseline
        .groupby(["season_year", "month"])["arithmetic_mean"]
        .agg(mean="mean", std="std", count="count", max="max")
        .reset_index()
    )

    month_climatology = (
        baseline
        .groupby("month")["arithmetic_mean"]
        .agg(clim_mean="mean", clim_std="std")
        .reset_index()
    )

    return month_climatology, monthly_baseline


def process_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise and label the event window data (Jan–Mar 2025).
    Hourly rows (88502) use 'sample_measurement'; daily rows (88101) use
    'arithmetic_mean'. Both are unified into a single 'pm25' column.
    """
    df = df.copy()
    df["date_local"] = pd.to_datetime(df["date_local"])

    # Unify value column across hourly (sample_measurement) and daily (arithmetic_mean)
    if "sample_measurement" in df.columns and "arithmetic_mean" in df.columns:
        df["pm25"] = df["sample_measurement"].combine_first(df["arithmetic_mean"])
    elif "sample_measurement" in df.columns:
        df["pm25"] = df["sample_measurement"]
    else:
        df["pm25"] = df["arithmetic_mean"]

    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["pm25"])

    def label_period(d):
        if d < pd.Timestamp("2025-01-07"):
            return "Before"
        elif d <= pd.Timestamp("2025-01-31"):
            return "During"
        else:
            return "After"

    df["period"] = df["date_local"].apply(label_period)
    return df


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(month_climatology: pd.DataFrame, event_window: pd.DataFrame):
    if not month_climatology.empty:
        print("\n" + "="*60)
        print("CLIMATOLOGY (2000–2024, Oct–Mar) — County-wide daily PM2.5 mean")
        print("="*60)
        clim_indexed = month_climatology.set_index("month")
        for m in FIRE_SEASON_MONTHS:
            if m not in clim_indexed.index:
                continue
            row   = clim_indexed.loc[m]
            mname = calendar.month_abbr[m]
            print(f"  {mname}  clim_mean={row['clim_mean']:.2f}  clim_std={row['clim_std']:.2f} µg/m³")

    if event_window.empty:
        return

    print("\n" + "="*60)
    print("EVENT WINDOW (Jan–Mar 2025) — PM2.5 by period")
    print("  Before : Jan 1  – Jan 6  2025  (pre-ignition)")
    print("  During : Jan 7  – Jan 31 2025  (active fire)")
    print("  After  : Feb 1  – Mar 31 2025  (post-containment)")
    print("="*60)
    period_summary = (
        event_window.groupby("period")["pm25"]
        .agg(mean="mean", std="std", max="max", n="count")
    )
    for period in ["Before", "During", "After"]:
        if period in period_summary.index:
            row = period_summary.loc[period]
            print(f"  {period:8s}  mean={row['mean']:.2f}  std={row['std']:.2f}  max={row['max']:.2f}  n={int(row['n'])}")

    if not month_climatology.empty:
        print("\n" + "="*60)
        print("ANOMALY — 2025 January vs historical climatology")
        print("="*60)
        jan_clim     = month_climatology[month_climatology["month"] == 1].iloc[0]
        jan_2025_mean = event_window[event_window["date_local"].dt.month == 1]["pm25"].mean()
        anomaly = jan_2025_mean - jan_clim["clim_mean"]
        sigma   = anomaly / jan_clim["clim_std"] if jan_clim["clim_std"] > 0 else float("nan")
        print(f"  Jan 2025 mean : {jan_2025_mean:.2f} µg/m³")
        print(f"  Climatology   : {jan_clim['clim_mean']:.2f} µg/m³")
        print(f"  Anomaly       : +{anomaly:.2f} µg/m³  ({sigma:+.1f}σ)")


# ── Save outputs ──────────────────────────────────────────────────────────────

def save(
    month_climatology: pd.DataFrame,
    monthly_baseline: pd.DataFrame,
    event_window: pd.DataFrame,
):
    print("\nSaved:")
    if PULL_HISTORICAL and not monthly_baseline.empty:
        month_climatology.to_csv("month_climatology_2000_2024.csv", index=False)
        monthly_baseline.to_csv("monthly_baseline_oct_mar_2000_2024.csv", index=False)
        print("  month_climatology_2000_2024.csv")
        print("  monthly_baseline_oct_mar_2000_2024.csv")
    if PULL_EVENT and not event_window.empty:
        event_window.to_csv("event_window_site_specific.csv", index=False)
        print("  event_window_site_specific.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not PULL_HISTORICAL and not PULL_EVENT:
        raise SystemExit("Both PULL_HISTORICAL and PULL_EVENT are False — nothing to do.")

    month_climatology = pd.DataFrame()
    monthly_baseline  = pd.DataFrame()
    event_window      = pd.DataFrame()

    if PULL_HISTORICAL:
        print("Fetching historical baseline (2000–2024, county-wide daily FRM)...")
        raw_historical = fetch_historical_data()
        print(f"  Rows fetched: {len(raw_historical):,}")
        month_climatology, monthly_baseline = process_historical(raw_historical)

    if PULL_EVENT:
        print("\nFetching event window (Jan–Mar 2025, per-site)...")
        raw_event = fetch_event_data()
        if not raw_event.empty:
            print(f"  Rows fetched: {len(raw_event):,}")
            event_window = process_event(raw_event)
        else:
            print("  No event data returned.")

    print_summary(month_climatology, event_window)
    save(month_climatology, monthly_baseline, event_window)

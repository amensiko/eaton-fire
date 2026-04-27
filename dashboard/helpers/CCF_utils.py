import pandas as pd
from functools import reduce
import yaml
#Utils to merge data sources from biodiversity observation, weather, news article, and air quality data.


def load_wx():
    wx_daily = r"helpers/data/weather/daily_data.csv"
    df = pd.read_csv(wx_daily)
    
    
    #Convert to datetime
    df["date"] = pd.to_datetime(df["DateString"], format="%Y%m%d")

    #drop extraneous
    df = df.drop(columns=["Year", "Month", "Day", "DateString"])
    print(df.head)
    return df

def load_aq(station=None):
    """
    loads air quality data, subsetted by station
    """
    #load processed data
    pm25_daily = "helpers/data/airquality/pm25_focal_stations_daily.csv"
    df = pd.read_csv(pm25_daily)
    #manage datetime
    df["date"] = pd.to_datetime(df['date'], format="%Y-%m-%d")

    #subset to station if requested, otherwise take the average across stations
    if station:
        #subset by station
        df = df[df['station']==station]
        df= df.drop(columns=["station", "label"]) #drop station column (filtered in function)
    else:
        df= df.groupby("date").mean(numeric_only=True).reset_index()
        df= df.drop(columns=["interpolated"]) #drop station column (filtered in function)

    return df

def load_biodiversity():

    bio_daily = r"helpers/data/biodiversity/biodiversity_daily_counts_cr.csv"
    df = pd.read_csv(bio_daily)
    
    
    #Convert to datetime
    df["date"] = pd.to_datetime(df["day"], format="%Y-%m-%d")

    df = df.drop(columns=["day"])


    return df

def load_news():
    news_daily = r"helpers/data/news/news_daily_counts_cr.csv"
    df = pd.read_csv(news_daily)
    #Convert to datetime
    df["date"] = pd.to_datetime(df["publish_date"], format="%Y-%m-%d")
    df = df.drop(columns=["publish_date"])


def merge_sources(station=None):
    """
    Merges air quality (subsetted by station), weather data sources (to add, biodiversity and news articles) - assumed to be on a daily timestep from 01/25 to 03/25
    """
    #initialize list to hold loaded data
    to_merge = []

    #load weather
    wx_data = load_wx()
    to_merge.append(wx_data)

    #load pm25
    pm25_data = load_aq(station) #load
    to_merge.append(pm25_data)

    bio_data = load_biodiversity()
    to_merge.append(bio_data)
    
    merged = reduce(lambda left, right: left.merge(right, on="date"), to_merge)

    return merged

def load_variable_lookup():
    var_yaml = r"helpers/data/variables.yaml"
    var_lookup_flat = {}
    
    with open(var_yaml, 'r') as f:
        var_lookup = yaml.safe_load(f)

    for var_type, dict in var_lookup['variables'].items():
        for key, val in dict.items():
            var_lookup_flat[key] = val
    
    return var_lookup_flat

#DateString can be kept, and year/month/day columns dropped for bio data.


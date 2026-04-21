import pandas as pd
from functools import reduce
import yaml
#Utils to merge data sources from biodiversity observation, weather, news article, and air quality data.
pm25_data = "aqs_data/pm25_focal_stations_daily.csv"
wx_data = "prism_wx/data/processed/daily_data.csv"

def load_wx():
    wx_daily = r"prism_wx/data/processed/daily_data.csv"
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
    pm25_daily = "aqs_data/pm25_focal_stations_daily.csv"
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
    
    merged = reduce(lambda left, right: left.merge(right, on="date"), to_merge)

    return merged

def load_variable_lookup():
    var_yaml = "variables.yaml"
    
    with open(var_yaml, 'r') as f:
        var_lookup = yaml.safe_load(f)
    
    return var_lookup

#DateString can be kept, and year/month/day columns dropped for wx data.


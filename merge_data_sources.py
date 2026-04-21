#Utils to merge data sources from biodiversity observation, weather, news article, and air quality data. 


#DateString can be kept, and year/month/day columns dropped for wx data.


#AQ station should probably be selected and/or averaged here, with 

def load_aqs():
  aqs_csv = r"pm25_focal_stations_daily.csv"
  df = pd.read_csv(aqs_csv, parse_dates=["date"]) #read csv

  #create column for each station
  df_aqs = (
      df.pivot(index="date", columns="station", values="pm25")
      .rename(columns=lambda s: f"aqs_{s}_pm25")
      .reset_index()
  )
  df_aqs.columns.name = None
  
  #create averae column
  df_aqs["aqs_avg_pm25"] = df_wide.filter(like="aqs_").mean(axis=1)

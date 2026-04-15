import requests
import argparse
import yaml
import pandas as pd
import numpy as np
import zipfile
import os
from io import BytesIO
from time import sleep



def daily_download(config):
    """
    Runs download process for daily data from PRISM

    Parameters: 
    - Config (dict): contains configuration details (general and daily-specific)
    """
    download_config = config["data_download"]["daily"]

    #Generate a list of dates for the whole range
    dates = pd.date_range(start=download_config["start_date"], end=download_config["end_date"])

    data_download(config=config,
                  timestep="daily",
                  dates=dates,
                  download_config=download_config
                  )

def monthly_download(config):
    """
    Runs download process for daily data from PRISM

    Parameters: 
    - Config (dict): contains configuration details (general and daily-specific)
    """
    download_config = config["data_download"]["monthly"]

    #Generate a list of dates for the whole range
    dates = []

    startyear = download_config["year_range"][0]
    endyear = download_config["year_range"][1]+1 #add 1 because zero indexing

    for year in np.arange(startyear, endyear, 1):
        for month in download_config["months"]:
            dates.append(f"{year}{month}")

    data_download(config=config,
                  timestep="monthly",
                  dates=dates,
                  download_config=download_config
                  )

    
def data_download(config, timestep, dates, download_config):
    """
    General download process for PRISM weather data

    Parameters: 
    - Config (dict): contains configuration details
    - timestep (str): Indicates whether daily or monthly data is being downloaded
    - dates (list): list of dates (YYYYMMDD or YYYYMM) for which data should be downloaded
    - download_config (dict): timestep-specific details from config (slightly redundant - made code less verbose)
    """
    for date in dates:
        if timestep == "daily":
            #formats as YYYYMMDD    
            date_schema = date.strftime('%Y%m%d')  # e.g., '200506'
        elif timestep == "monthly":
            date_schema = date

        for wx_var in download_config["wx_vars"]:
            #dynamically generate url
            url = f"{config["data_download"]["base_url"]}/{config["data_download"]["region"]}/{config["data_download"]["resolution"]}/{wx_var}/{date_schema}"

            #dynamically generated output path
            out_path = f"{download_config["save_loc"]}/{date_schema}_{wx_var}"
            
            #Don't make the request if the file exists
            if os.path.exists(out_path):
                print(f"Path exists... skipped {out_path}")  
                continue

            #Request
            response = requests.get(url)
            response.raise_for_status()
            print(f"  Status: {response.status_code}, Size: {len(response.content)} bytes")

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(out_path)
            
            
            #be respectful with requests (per an example on PRISMS site)
            sleep(2)
             


def main():
    #parse inputs from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("timestep")
    args = parser.parse_args()

    #open the YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    timestep = args.timestep
    
    if timestep == "monthly":
        print("Downloading monthly data")
        monthly_download(config)
    else:
        print("Downloading daily data")
        daily_download(config)


if __name__ == "__main__":
   main()
import requests
import argparse
import yaml
import pandas as pd
import zipfile
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

    for year in download_config["year_range"]:
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
            url = f"{config["base_url"]}/{config["region"]}/{config["resolution"]}/{wx_var}/{date_schema}"

            #Request
            response = requests.get(url)
            response.raise_for_status()
            print(f"  Status: {response.status_code}, Size: {len(response.content)} bytes")

            #dynamically generated output path
            out_path = f"{download_config["save_loc"]}/{date_schema}_{wx_var}"

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(out_path)

            #be respectful with requests (per an example on PRISMS site)
            sleep(2)        


def main():
    #parse inputs from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    #open the YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    #daily_download(config)
    monthly_download(config)


if __name__ == "__main__":
   main()
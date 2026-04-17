import argparse
import yaml
import rasterio
from rasterio.mask import mask
import pandas as pd
import numpy as np
import geopandas as gpd


def construct_tif_path(config, timestep, datestring, wx_var):
    """
    Builds a filepath to the PRISM standardized filename for the appropriate .tif
    """
    resolution_s = config["data_download"]["resolution_s"]
    filepath = config["data_download"][f"{timestep}"]["save_loc"]
    tif_path = f"{filepath}/{datestring}_{wx_var}/prism_{wx_var}_us_{resolution_s}_{datestring}.tif"
    
    return tif_path

def extract_AOI_average(config, timestep, aoi, datestring, wx_var):
    """
    Extracts the average pixel value for cells overlapping a provided AOI
    """
    tif_path = construct_tif_path(config, timestep, datestring, wx_var)
    aoi_geom = aoi.geometry.values[0]

    #Open the tif and get the average of the variable in AOI:
    with rasterio.open(tif_path) as src:
        #mask to aoi
        masked_array, _ = mask(src, [aoi_geom], crop=True)
        
        #Do not count nodata's
        nodata_val = src.nodata
        data = masked_array[0]
        data[data == nodata_val] = np.nan
        #Find mean in aoi
        mean_value = np.nanmean(masked_array[0])
        
        return mean_value
    

def construct_lookback_df(config, timestep):
    
    #rows for making a dataframe
    rows = []

    if timestep == "monthly":
        #access the config
        monthly_config = config["data_download"]["monthly"]

        #Get the start and end years
        startyear = monthly_config["year_range"][0]
        endyear = monthly_config["year_range"][1]+1 #add 1 because zero indexing

        #loop through all requested dates
        for year in np.arange(startyear, endyear, 1):
            for month in monthly_config["months"]:
                #populate a list of the dates
                date_string =f"{year}{month}" 
                #Add date data as a dictionary to the list of rows
                rows.append(
                    {
                        "DateString" : date_string,
                        "Year" : year,
                        "Month" : month
                    }
                )

    elif timestep == "daily":
        #access daily config
        daily_config = config["data_download"]["daily"]
        #get list of all dates
        dates = pd.date_range(start=daily_config["start_date"], end=daily_config["end_date"])
        #loop over them
        for date in dates:
            #date data dicts as rows in row list
            date_string = date.strftime('%Y%m%d')
            rows.append(
                    {
                        "DateString" : date_string,
                        "Year" : date_string[:4],
                        "Month" : date_string[4:6],
                        "Day" : date_string[6:]
                    }
                )


    #Now populate the dataframe from rows 
    wx_lookback = pd.DataFrame(rows)

    return wx_lookback
    
    
def populate_lookback_vars(config, timestep, aoi, lookback_df):

    timestep_config = config["data_download"][f"{timestep}"]
    
    for wx_var in timestep_config["wx_vars"]:
        means = []
        for idx, row in lookback_df.iterrows():
            datestring = row["DateString"]
            mean_val = extract_AOI_average(config, timestep, aoi, datestring, wx_var)
            means.append(mean_val)
        
        lookback_df[wx_var] = means
    
    return



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

    df = construct_lookback_df(config, timestep)
    
    #load aoi
    aoi = gpd.read_file(config["data_download"]["aoi"])
    aoi_reproj = aoi.to_crs("EPSG:4269")

    #Extract weather variables from raw data in place
    populate_lookback_vars(config, timestep, aoi, df)
    
    if timestep == "monthly":
        df.to_csv(config["monthly_lookback_data"]["path"])
    elif timestep =="daily":
        df.to_csv(config["daily_data"]["path"])
        print(df.shape)
        print(df.head())

    


if __name__ == "__main__":
    main()
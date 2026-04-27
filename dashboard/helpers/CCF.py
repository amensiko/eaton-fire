from statsmodels.tsa.stattools import ccf
import pandas as pd
import plotly.graph_objects as go
# from CCF_utils import merge_sources, load_variable_lookup
from helpers.CCF_utils import merge_sources, load_variable_lookup

#One function to calculate, one function to execute
def compute_cross_correlation(df, var1, var2):
    """
    Computes temporal cross correlations and 95% confidence intervals for two variables

    Parameters: 

    df (pd.dataframe): dataframe with daily data for all variables of interest
    var1 (str): "driver" variable for cross correlation - must exist in df
    var2 (str): "lagged" variable for cross correlation - must exist in df
    
    """

    cross_correlation = ccf(x=df[var1], #x shifts in time, ie, for lag k, the data is shifted to start at x(t+k)
                            y=df[var2], #y stays in place
                            nlags=25, #More than this and things start to get unreliable (less data) - are there domain reasons to go beyond?
                            adjusted=True, # adjust for decreasing number of samples at higher lags
                            alpha=0.05 #95% confidence interval
                            )
    
    return cross_correlation

def plot_correlations(cross_corr, var1, var2):
    correlations = cross_corr[0]
    confidence_ints = cross_corr[1]
    lags = [i for i in range(len(correlations))]

    fig = go.Figure(
        go.Bar(
            x=lags,
            y=correlations,
            error_y=dict(
                type="data",
                symmetric=False,
                array=confidence_ints[:, 1] - correlations,
                arrayminus=correlations - confidence_ints[:, 0],
            ),
        )
    )

    var_lookup = load_variable_lookup()
    var1_label = var_lookup[var1]
    var2_label = var_lookup[var2]

    fig.update_layout(
        title=f"Cross-Correlation Plot: {var1_label} & {var2_label}",
        template="plotly_white",
        height=600,
    )
    fig.update_yaxes(title_text="Correlation")
    fig.update_xaxes(title_text="Lag (days)")

    return fig

def cross_correlation(var1, var2, station=None):
    df = merge_sources(station)
    cross_corr = compute_cross_correlation(df, var1, var2)
    fig = plot_correlations(cross_corr, var1, var2)
    return fig
 

def main():
    """
    selection of variables and weather station
    """

    #Variables to run CCF on - see variables.yaml
    var1 = "ppt"
    var2 = "pm25"

    #Stations to select from - right now you HAVE to choose a station
    stations = ["Glendora",
                "Los Angeles-North Main Street",
                "North Hollywood (NOHO)",
                "Pasadena"
                ]
    
    #select station
    station = stations[2]

    cross_correlation(var1="ppt", #Select x (driver) variable
                      var2="pm25", #Select y (lagged) variable
                      station=None #select station for PM25 (or not - will average)
                      )
    


if __name__ == "__main__":
    main()
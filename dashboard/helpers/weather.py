import argparse
import yaml
import pandas as pd
import pymannkendall as mk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def add_month_names(df):
    month_lookup = {
    10: "October",
    11: "November",
    12: "December"
  }
    df["Month_Name"] = df["Month"].map(month_lookup)


def load_lookback():
    loaded = pd.read_csv("helpers/data/weather/monthly_lookback.csv")
    add_month_names(loaded)
    return loaded

def construct_multipanel_bars(wx_var):
    """
    Constructs a multipanel figure for each wx variable, with each panel showing a bar chart/mann kendall trend test.
    """
    #load data
    df = load_lookback()

    #months information
    month_list = [10, 11, 12]

    month_names = {
                    10: "October",
                    11: "November",
                    12: "December"
                }
    
    variable_names = {
                        "vpdmax": "Vapor Pressure Deficit (Mean Daily Maximum, hPa)",
                        "ppt": "Precipitation (mm)",
                        "tmean": "Temperature (Daily Mean, C)",
                    }
    
    variable_name = variable_names[wx_var]
        
    subplot_titles = tuple(month_names[m] for m in month_list)
    fig = make_subplots(rows=1, cols=len(month_list), subplot_titles=subplot_titles)

    for i, month in enumerate(month_list, 1):
        month_df = df[df['Month'] == month].sort_values('Year')
        month_name = month_names[month]

        fig.add_trace(go.Bar(x=month_df['Year'], y=month_df[wx_var],
                                name=month_name, showlegend=False),
                        row=1, col=i)

        # Mann-Kendall
        result = mk.original_test(month_df[wx_var])
        trend = result.trend.capitalize()
        tau = result.Tau
        p = result.p

        x_center = month_df['Year'].median()

        fig.add_annotation(
            x=x_center,
            xref=f"x{i}",
            y=0,
            yref="paper",
            yshift=-50,
            text=f"{trend}<br>τ={tau:.2f}, p={p:.3f}",
            showarrow=False,
        )

    fig.update_layout(
        title_text=f'{variable_name} by Month',
        margin=dict(b=100)
    )
    fig.update_yaxes(title_text=variable_name, row=1, col=1)

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    return fig

def construct_multipanel_box(wx_var):
    """
    Constructs multipanel boxplots (one panel per month) for each wx variable listed in config.

    Parameters:
        - config: project configuration from yaml (parsed from command line)
        - df: a dataframe containing wide format time series data for each weather variable (monthly). Should be in  
    """

    #load data
    df = load_lookback()

    #month info
    month_list = [10, 11, 12]

    month_names = {
                    10: "October",
                    11: "November",
                    12: "December"
                }
    
    variable_names = {
                        "vpdmax": "Vapor Pressure Deficit (Mean Daily Maximum, hPa)",
                        "ppt": "Precipitation (mm)",
                        "tmean": "Temperature (Daily Mean, C)",
                    }
    
    
    #BOX PLOTS FOR ALL VARIABLES BY MONTH
    subplot_titles = tuple(month_names[m] for m in month_list)
    fig = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles)
    
    #make Monthly subplots
    for i, month in enumerate(month_list, 1):
        #access the data for the given month
        month_data = df[df['Month'] == month][wx_var]
        month_name = month_names[month]
        variable_name = variable_names[wx_var]
        
        fig.add_trace(go.Box(y=month_data, name=month_name, showlegend=False),
                    row=1, col=i)
        
        #add a point for 2024 on each
        val_2024 = df[(df['Year'] == 2024) & (df['Month'] == month)][wx_var].values[0]

        #compute z-score for the 2024 point
        z = (val_2024 - month_data.mean()) / month_data.std()
        
        fig.add_trace(go.Scatter(x=[month_name], 
                                    y=[val_2024],
                                    mode='markers',
                                    marker=dict(size=12, color='red'),
                                    name='2024', showlegend=(i==1)),
                                    row=1,
                                    col=i)
        
        #annotate the 2024 point
        fig.add_annotation(
            x=month_name,
            xref=f"x{i}",
            y=0,
            yref="paper",
            yshift=-40,
            text=f"2024 z-score: {z:.2f}",
            showarrow=False,
        )
    
    # Calculate min/max across all months for this variable
    y_min = df[df['Month'].isin(month_list)][wx_var].min()
    y_max = df[df['Month'].isin(month_list)][wx_var].max()

    fig.update_yaxes(range=[y_min, y_max])

    
    fig.update_layout(title_text=f'{variable_name} Distribution by Month')
    fig.update_yaxes(title_text=variable_name, row=1, col=1)

    fig.update_layout(
        template="plotly_white",
        height=450
    )
    
    return fig



def main():
    construct_multipanel_bars("ppt")
    construct_multipanel_box("ppt")

if __name__ == "__main__":
    main()
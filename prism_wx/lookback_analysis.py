import argparse
import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px



def add_month_names(config, df):
    month_lookup = config["monthly_lookback_data"]["month_names"]
    df["Month_Name"] = df["Month"].map(month_lookup)

def load_lookback(config):
    loaded = pd.read_csv(config["monthly_lookback_data"]['path'])
    add_month_names(config, loaded)
    return loaded

def construct_multipanel_bars(config, df):

    for var in config["data_download"]["monthly"]["wx_vars"]:
        fig = px.bar(df, x='Year', y=var, facet_col='Month_Name',
                    title=f'Monthly {var} by Month')
        fig.show()

def construct_multipanel_box(config, df):

    config_month_names = config['monthly_lookback_data']['month_names']
    config_var_names = config['monthly_lookback_data']['variable_names']
    month_list = [int(m) for m in config['data_download']['monthly']['months']]
    variables = config['data_download']['monthly']['wx_vars']

    for var in variables:
        #BOX PLOTS FOR ALL VARIABLES BY MONTH
        subplot_titles = tuple(config_month_names[m] for m in month_list)
        fig = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles)
        
        #make Monthly subplots
        for i, month in enumerate(month_list, 1):
            #access the data for the given month
            month_data = df[df['Month'] == month][var]
            month_name = config_month_names[month]
            variable_name = config_var_names[var]
            fig.add_trace(go.Box(y=month_data, name=month_name, showlegend=False),
                        row=1, col=i)
            
            #add a point for 2024 on each
            val_2024 = df[(df['Year'] == 2024) & (df['Month'] == month)][var].values[0]
            
            fig.add_trace(go.Scatter(x=[month_name], 
                                     y=[val_2024],
                                     mode='markers',
                                     marker=dict(size=12, color='red'),
                                     name='2024', showlegend=(i==1)),
                                     row=1,
                                     col=i)
        
        # Calculate min/max across all months for this variable
        y_min = df[df['Month'].isin(month_list)][var].min()
        y_max = df[df['Month'].isin(month_list)][var].max()

        fig.update_yaxes(range=[y_min, y_max])

        
        fig.update_layout(title_text=f'{variable_name} Distribution by Month')
        fig.update_yaxes(title_text=variable_name, row=1, col=1)
        
        #Save as PNG
        output_path = config['monthly_lookback_data']['figures']
        fig.write_image(f"{output_path}/{var}_lookback_distribution.png")





def main():
    #parse inputs from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    #open the YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    df = load_lookback(config)
    #construct_multipanel_bars(config, df)
    construct_multipanel_box(config, df)

if __name__ == "__main__":
    main()